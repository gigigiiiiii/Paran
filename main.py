from collision_monitor.config import parse_args
from collision_monitor.runner import PipelineRunner


def main():
    args = parse_args()
    runner = PipelineRunner(args)
    runner.run()


if __name__ == "__main__":
    main()

