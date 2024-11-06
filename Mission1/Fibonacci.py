febo = []
febo.append(1)
febo.append(1)

n = int(input())
# print(n)

for i in range(2, n):
    febo.append(febo[i - 1] + febo[i - 2])

for i in range(n):
    print(febo[i])


def fibonacci_generator():
    # 初始值，斐波那契数列的前两个数字是 1 和 1
    a, b = 1, 1

    # 无限循环，生成斐波那契数列的每一个数
    while True:
        # 使用 yield 暂停函数并返回当前的 a 值
        yield a

        # 计算下一个斐波那契数列的值
        # a = b, b = a + b
        # 例如最开始：a = 1, b = 1, 下一轮变为 a = 1, b = 2
        # 然后：a = 2, b = 3, 依此类推
        a, b = b, a + b


# 创建生成器对象
gen = fibonacci_generator()

# 输出前10个斐波那契数列的值
for _ in range(10):
    print(next(gen))
