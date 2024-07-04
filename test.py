seg_outputs = [1,2,3,0,3,2,3]
x = tuple([seg_outputs[i] for i in range(len(seg_outputs) - 1, -1, -1)])
print(x)