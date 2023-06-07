from alrd.agent.absagent import Agent

class RepeatAgent(Agent):
    def __init__(self, agent, repeat) -> None:
        self.agent = agent
        self.repeat = repeat
        self.last_action = None
        self.count = 0

    def act(self, obs):
        if self.count % self.repeat == 0:
            action = self.agent.act(obs)
        else:
            action = self.last_action
        self.last_action = action
        self.count += 1
        return action

    def reset(self):
        self.agent.reset()
        self.count = 0