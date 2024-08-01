def BinarySearch(nums, target):
    start = 0
    end = len(nums) - 1
    while start <= end:
        mid = (start + end) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            start = mid + 1
        else:
            end = mid - 1
    return -1
if __name__ == "__main__":
    nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    target = 5
    print(BinarySearch(nums, target))
    nums = [1, 2, 3, 4, 6, 7, 8, 9, 10]
    print(BinarySearch(nums, target))
    