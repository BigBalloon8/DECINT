import argparse
def Decint_ai_node_arg_parser():
    parser = argparse.ArgumentParser(description='For use by automated AI nodes NOT users')
    #args dont have description as not for users to use
    parser.add_argument("--rank", dest="my_rank")
    parser.add_argument("--size", dest="size")
    parser.add_argument("--address", dest="my_address")
    parser.add_argument("--port", dest="my_port")
    return parser