nums = [1,2,3,4,5,6]
oddNums = [x for x in nums if x % 2 == 1]
print oddNums
oddNumsPlusOne = [x+1 for x in nums if x % 2 ==1]
print oddNumsPlusOne

# strings = ['SHORT','LONG','VERYLONG']
# newStrings = [x.lower() for x in strings if len(x)>5]
# print newStrings