def automaton(word):
    state = "q0"

    for symbol in word:

        if state == "q0":
            if symbol == "a":
                state = "q1"
            elif symbol == "b":
                state = "q2"
            else:
                return False

        elif state == "q1":
            if symbol == "a":
                state = "q1"
            elif symbol == "b":
                state = "q2"
            else:
                return False

        elif state == "q2":
            if symbol == "b":
                state = "q2"
            elif symbol == "a":
                return False
            else:
                return False

    return state == "q1" or state == "q2"


word = input("Введите слово: ")

if automaton(word):
    print("Слово принадлежит языку")
else:
    print("Слово не принадлежит языку")