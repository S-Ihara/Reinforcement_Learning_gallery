

def check_env_stats(env):
    """環境のステータスを確認する
    args:
        env (gymnasium.Env): 環境
    """
    print(f"observation_space: {env.observation_space}")
    print(f"action_space: {env.action_space}")
    print(f"spec: {env.spec}")
