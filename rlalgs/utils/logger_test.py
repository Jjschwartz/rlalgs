from rlalgs.utils.logger import Logger


if __name__ == "__main__":
    logger = Logger()

    for i in range(10):
        logger.log_tabular("tInt", i)
        logger.log_tabular("tFloat", 1.0 * i)
        logger.dump_tabular()
        input()
