import logging


logging.basicConfig(filename="log.txt", level=logging.DEBUG,
            format=f"%(levelname)-8s %(asctime)s \t %(filename)s @function %(funcName)s line %(lineno)s - %(message)s",
            filemode="w")


