import time

# Параметры варианта 1
DEPTH = 5
WIDTH = 2

# 32 фиксированных значения листьев
LEAVES = [
    3, 5, 2, 9, 1, 7, 4, 6,
    8, 0, 10, 2, 5, 3, 7, 1,
    9, 4, 6, 8, 2, 5, 7, 3,
    1, 10, 4, 6, 8, 9, 2, 7
]

# Индекс для распределения листьев
leaf_index = 0


# Создание дерева

def generate_tree(depth):
    global leaf_index

    if depth == 0:
        value = LEAVES[leaf_index]
        leaf_index += 1
        return value

    return [
        generate_tree(depth - 1)
        for _ in range(WIDTH)
    ]


# Обычный Mini-Max

minimax_nodes = 0


def minimax(node, depth, maximizing_player):
    global minimax_nodes

    minimax_nodes += 1

    # Дошли до листа
    if depth == 0:
        return node

    # Ход MAX
    if maximizing_player:
        best_value = float("-inf")

        for child in node:
            value = minimax(
                child,
                depth - 1,
                False
            )

            best_value = max(best_value, value)

        return best_value

    # Ход MIN
    else:
        best_value = float("inf")

        for child in node:
            value = minimax(
                child,
                depth - 1,
                True
            )

            best_value = min(best_value, value)

        return best_value


# Mini-Max с Alpha-Beta отсечением

alphabeta_nodes = 0
alphabeta_cutoffs = 0


def alphabeta(node, depth, alpha, beta, maximizing_player):
    global alphabeta_nodes
    global alphabeta_cutoffs

    alphabeta_nodes += 1

    # Дошли до листа
    if depth == 0:
        return node

    # Ход MAX
    if maximizing_player:
        best_value = float("-inf")

        for child in node:
            value = alphabeta(
                child,
                depth - 1,
                alpha,
                beta,
                False
            )

            best_value = max(best_value, value)
            alpha = max(alpha, best_value)

            # Alpha-Beta отсечение
            if beta <= alpha:
                alphabeta_cutoffs += 1
                break

        return best_value

    # Ход MIN
    else:
        best_value = float("inf")

        for child in node:
            value = alphabeta(
                child,
                depth - 1,
                alpha,
                beta,
                True
            )

            best_value = min(best_value, value)
            beta = min(beta, best_value)

            # Alpha-Beta отсечение
            if beta <= alpha:
                alphabeta_cutoffs += 1
                break

        return best_value


# Основная программа

print("МИНИ-МАКС С АЛЬФА-БЕТА ОТСЕЧЕНИЕМ")
print("Вариант 1")

print(f"Глубина дерева: {DEPTH}")
print(f"Ширина дерева: {WIDTH}")
print("Значения листьев: фиксированные")
print(f"Количество листьев: {len(LEAVES)}")

# Создаём дерево
leaf_index = 0
tree = generate_tree(DEPTH)


# Mini-Max

start = time.perf_counter()

minimax_result = minimax(
    tree,
    DEPTH,
    True
)

minimax_time = time.perf_counter() - start

# Alpha-Beta

start = time.perf_counter()

alphabeta_result = alphabeta(
    tree,
    DEPTH,
    float("-inf"),
    float("inf"),
    True
)

alphabeta_time = time.perf_counter() - start

# Вывод результатов

print("\nРЕЗУЛЬТАТЫ")

print("Обычный Mini-Max:")
print(f"  Результат: {minimax_result}")
print(f"  Проверено узлов: {minimax_nodes}")
print(f"  Время: {minimax_time:.8f} секунд")

print("\nMini-Max с Alpha-Beta:")
print(f"  Результат: {alphabeta_result}")
print(f"  Проверено узлов: {alphabeta_nodes}")
print(f"  Время: {alphabeta_time:.8f} секунд")
print(f"  Количество отсечений: {alphabeta_cutoffs}")


# Сравнение

print("\nСРАВНЕНИЕ")

if minimax_result == alphabeta_result:
    print("Результаты совпадают: ДА")
else:
    print("Результаты совпадают: НЕТ")


if minimax_nodes > 0:
    reduction = (
        (minimax_nodes - alphabeta_nodes)
        / minimax_nodes
        * 100
    )

    print(
        f"Сокращение проверенных узлов: "
        f"{reduction:.2f}%"
    )


if alphabeta_time < minimax_time:
    print("Alpha-Beta работает быстрее: ДА")
else:
    print("Alpha-Beta работает быстрее: НЕТ")
