def concat_cmd(cmd, *args):
    result = cmd
    for arg in args:
       result = str(result) + ',' + str(arg)
#    for arg in args:
#        result.append(str(arg))
    return result
