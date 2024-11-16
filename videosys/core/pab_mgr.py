from videosys.utils.logging import logger

PAB_MANAGER = None


class PABConfig:
    def __init__(
        self,
        # 执行交叉广播开关
        cross_broadcast: bool = False,
        # 用于空间广播的时间步范围，哪些时间步会考虑进行空间广播
        cross_threshold: list = None,
        # 交叉广播的时间步范围
        cross_range: int = None,
        # 空间广播
        spatial_broadcast: bool = False,
        spatial_threshold: list = None,
        spatial_range: int = None,
        # 时间广播
        temporal_broadcast: bool = False,
        temporal_threshold: list = None,
        temporal_range: int = None,
        # mlp是一个前馈神经网络，与self attention交叉使用
        mlp_broadcast: bool = False,
        mlp_spatial_broadcast_config: dict = None,
        mlp_temporal_broadcast_config: dict = None,
    ):
        self.steps = 24
        self.cross_broadcast = cross_broadcast
        self.cross_threshold = cross_threshold
        self.cross_range = cross_range

        self.spatial_broadcast = spatial_broadcast
        self.spatial_threshold = spatial_threshold
        self.spatial_range = spatial_range

        self.temporal_broadcast = temporal_broadcast
        self.temporal_threshold = temporal_threshold
        self.temporal_range = temporal_range

        self.mlp_broadcast = mlp_broadcast
        self.mlp_spatial_broadcast_config = mlp_spatial_broadcast_config
        self.mlp_temporal_broadcast_config = mlp_temporal_broadcast_config
        self.mlp_temporal_outputs = {}
        self.mlp_spatial_outputs = {}


class PABManager:
    def __init__(self, config: PABConfig):
        self.config: PABConfig = config
        # 把所有的配置参数写入日志
        init_prompt = f"Init Pyramid Attention Broadcast."
        init_prompt += f" spatial broadcast: {config.spatial_broadcast}, spatial range: {config.spatial_range}, spatial threshold: {config.spatial_threshold}."
        init_prompt += f" temporal broadcast: {config.temporal_broadcast}, temporal range: {config.temporal_range}, temporal_threshold: {config.temporal_threshold}."
        init_prompt += f" cross broadcast: {config.cross_broadcast}, cross range: {config.cross_range}, cross threshold: {config.cross_threshold}."
        init_prompt += f" mlp broadcast: {config.mlp_broadcast}."
        logger.info(init_prompt)

    def if_broadcast_cross(self, timestep: int, count: int):
        if (
            self.config.cross_broadcast
            and (timestep is not None)
            # 只有在 count 不是 cross_range 的倍数时才会触发广播,比如说369步算出的注意力，会作为后面几步的注意力
            and (count % self.config.cross_range != 0)
            and (self.config.cross_threshold[0] < timestep < self.config.cross_threshold[1])
        ):
            flag = True
        else:
            flag = False
        # 每次调用函数时，count 增加 1，并通过取模运算限制其值不超过总共要算的注意力步数。当达到 steps 后，count 会重置为 0。
        count = (count + 1) % self.config.steps
        return flag, count

    def if_broadcast_temporal(self, timestep: int, count: int):
        if (
            self.config.temporal_broadcast
            and (timestep is not None)
            and (count % self.config.temporal_range != 0)
            and (self.config.temporal_threshold[0] < timestep < self.config.temporal_threshold[1])
        ):
            flag = True
        else:
            flag = False
        count = (count + 1) % self.config.steps
        return flag, count

    def if_broadcast_spatial(self, timestep: int, count: int):
        if (
            self.config.spatial_broadcast
            and (timestep is not None)
            and (count % self.config.spatial_range != 0)
            and (self.config.spatial_threshold[0] < timestep < self.config.spatial_threshold[1])
        ):
            flag = True
        else:
            flag = False
        count = (count + 1) % self.config.steps
        return flag, count

    @staticmethod
    def _is_t_in_skip_config(all_timesteps, timestep, config):
        is_t_in_skip_config = False
        skip_range = None
        for key in config:
            if key not in all_timesteps:
                continue
            index = all_timesteps.index(key)
            skip_range = all_timesteps[index : index + 1 + int(config[key]["skip_count"])]
            if timestep in skip_range:
                is_t_in_skip_config = True
                skip_range = [all_timesteps[index], all_timesteps[index + int(config[key]["skip_count"])]]
                break
        return is_t_in_skip_config, skip_range

    def if_skip_mlp(self, timestep: int, count: int, block_idx: int, all_timesteps, is_temporal=False):
        if not self.config.mlp_broadcast:
            return False, None, False, None

        if is_temporal:
            cur_config = self.config.mlp_temporal_broadcast_config
        else:
            cur_config = self.config.mlp_spatial_broadcast_config

        is_t_in_skip_config, skip_range = self._is_t_in_skip_config(all_timesteps, timestep, cur_config)
        next_flag = False
        if (
            self.config.mlp_broadcast
            and (timestep is not None)
            and (timestep in cur_config)
            and (block_idx in cur_config[timestep]["block"])
        ):
            flag = False
            next_flag = True
            count = count + 1
        elif (
            self.config.mlp_broadcast
            and (timestep is not None)
            and (is_t_in_skip_config)
            and (block_idx in cur_config[skip_range[0]]["block"])
        ):
            flag = True
            count = 0
        else:
            flag = False

        return flag, count, next_flag, skip_range

    def save_skip_output(self, timestep, block_idx, ff_output, is_temporal=False):
        if is_temporal:
            self.config.mlp_temporal_outputs[(timestep, block_idx)] = ff_output
        else:
            self.config.mlp_spatial_outputs[(timestep, block_idx)] = ff_output

    def get_mlp_output(self, skip_range, timestep, block_idx, is_temporal=False):
        skip_start_t = skip_range[0]
        if is_temporal:
            skip_output = (
                self.config.mlp_temporal_outputs.get((skip_start_t, block_idx), None)
                if self.config.mlp_temporal_outputs is not None
                else None
            )
        else:
            skip_output = (
                self.config.mlp_spatial_outputs.get((skip_start_t, block_idx), None)
                if self.config.mlp_spatial_outputs is not None
                else None
            )

        if skip_output is not None:
            if timestep == skip_range[-1]:
                # TODO: save memory
                if is_temporal:
                    del self.config.mlp_temporal_outputs[(skip_start_t, block_idx)]
                else:
                    del self.config.mlp_spatial_outputs[(skip_start_t, block_idx)]
        else:
            raise ValueError(
                f"No stored MLP output found | t {timestep} |[{skip_range[0]}, {skip_range[-1]}] | block {block_idx}"
            )

        return skip_output

    def get_spatial_mlp_outputs(self):
        return self.config.mlp_spatial_outputs

    def get_temporal_mlp_outputs(self):
        return self.config.mlp_temporal_outputs


def set_pab_manager(config: PABConfig):
    global PAB_MANAGER
    PAB_MANAGER = PABManager(config)


def enable_pab():
    if PAB_MANAGER is None:
        return False
    return (
        PAB_MANAGER.config.cross_broadcast
        or PAB_MANAGER.config.spatial_broadcast
        or PAB_MANAGER.config.temporal_broadcast
    )


def update_steps(steps: int):
    if PAB_MANAGER is not None:
        PAB_MANAGER.config.steps = steps


def if_broadcast_cross(timestep: int, count: int):
    if not enable_pab():
        return False, count
    return PAB_MANAGER.if_broadcast_cross(timestep, count)


def if_broadcast_temporal(timestep: int, count: int):
    if not enable_pab():
        return False, count
    return PAB_MANAGER.if_broadcast_temporal(timestep, count)


def if_broadcast_spatial(timestep: int, count: int):
    if not enable_pab():
        return False, count
    return PAB_MANAGER.if_broadcast_spatial(timestep, count)


def if_broadcast_mlp(timestep: int, count: int, block_idx: int, all_timesteps, is_temporal=False):
    if not enable_pab():
        return False, count
    return PAB_MANAGER.if_skip_mlp(timestep, count, block_idx, all_timesteps, is_temporal)


def save_mlp_output(timestep: int, block_idx: int, ff_output, is_temporal=False):
    return PAB_MANAGER.save_skip_output(timestep, block_idx, ff_output, is_temporal)


def get_mlp_output(skip_range, timestep, block_idx: int, is_temporal=False):
    return PAB_MANAGER.get_mlp_output(skip_range, timestep, block_idx, is_temporal)
