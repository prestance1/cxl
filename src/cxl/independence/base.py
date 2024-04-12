import abc


class IndependenceTester(abc.ABC):

    def is_independent():
        pass


class ConditionalIndependenceTester(IndependenceTester):

    def is_conditionally_independent():
        pass
