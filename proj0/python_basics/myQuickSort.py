from random import randint

def sort(lst):
	if len(lst)==0:
		return []
	pivot = lst[0]
	front = [x for x in lst[1:] if x<pivot]
	back = [x for x in lst[1:] if x>=pivot]
	lst = sort(front) + [pivot] + sort(back)
	return lst

# Main Method
if __name__ == '__main__':    
    lst = []
    for i in range(10):
    	lst.append(randint(0,9))
    print lst
    print sort(lst)