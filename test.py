def say_joke(num = 1000):   

    print(f'def is_even(num):')
    for i in range(num):
        print(f'    if (num == {i}):')
        print(f'        return {True if (i % 2 == 0) else False}')
    print(f'    else:')
    print(f'        return num % 2 == 0 # edge case')