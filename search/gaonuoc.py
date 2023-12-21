import math


# Hàm heuristic đánh giá
def h(state):
    return 2 * math.ceil((m - state[2]) / max(inp))


def AStar():
    # Số bước đã thực hiện, Trạng thái các gáo nước (đầu tiền là rỗng), trạng thái bể, các bước thực hiện
    begin_state = (0, [0 for _ in inp], 0, [])

    heap = [begin_state]

    # A-star
    while len(heap) > 0:
        # Cập nhật lại heap theo (số bước đã thực hiện + giá trị heuristic)
        heap.sort(key=lambda e: e[0] + h(e))
        state = heap.pop(0)

        step_num = state[0]
        cans = state[1]
        current = state[2]
        trace = state[3]

        # if step_num == 3:
        #     continue
        print(state)

        if current == m:
            return trace

        # Đổ nước từ sông vào gáo
        for i, can in enumerate(cans):
            # Nếu đã đầy nước rồi thì không cần đổ lại
            if can == inp[i]:
                continue
            _cans = cans[:]
            _cans[i] = inp[i]
            next_trace = trace[:]

            # gáo -2: sông
            next_trace.append((-2, i))
            heap.append((step_num + 1, _cans, current, next_trace))

        # Đổ nước vào bể, nếu gáo chứa nước
        for i, can in enumerate(cans):
            if can == 0:
                continue

            if current + can > m:
                continue

            val = cans[i]
            _cans = cans[:]
            _cans[i] = 0
            next_trace = trace[:]

            # gáo -1: bể
            next_trace.append((i, -1))
            heap.append((step_num + 1, _cans, current + val, next_trace))

        # Đổ nước từ can này sang can khác (i sang j)
        for i, can1 in enumerate(cans):
            for j, can2 in enumerate(cans):
                if i == j:
                    continue
                # Không có nước thì không cần đổ
                if can1 == 0:
                    continue
                if can2 == inp[j]:
                    continue

                # Bỏ đi trường hợp đổ qua đổ lại
                last_can1, last_can2 = trace[-1]
                if i == last_can2 and j == last_can1:
                    continue

                # Đổ hết can1 qua can2 hoặc đổ đầy can2 và can1 vẫn còn nước
                dif = min([can1, inp[j] - can2])

                _cans = cans[:]

                _cans[i] -= dif
                _cans[j] += dif

                next_trace = trace[:]
                next_trace.append((i, j))
                heap.append((step_num + 1, _cans, current, next_trace))

        # Đổ đi can nước -> đổ vào sông
        for i, can in enumerate(cans):
            # Nếu đã trống thì không cần đổ
            if can == 0:
                continue

            a, b = trace[-1]
            # Múc đi đổ lại
            if a == -2 and b == i:
                continue
            _cans = cans[:]
            _cans[i] = 0
            next_trace = trace[:]

            # gáo -2: sông
            next_trace.append((i, -2))
            heap.append((step_num + 1, _cans, current, next_trace))
    return False


if __name__ == "__main__":
    inp = list(map(int, input().split()))
    n = inp[0]
    m = inp[1]
    inp = inp[2:]

    result = AStar()
    if not result:
        print("Không có đáp án")
    else:
        state = [0 for _ in inp]
        current = 0
        for step in result:
            a, b = step
            an = "bờ sông" if a == -2 else "bể" if a == -1 else f"gáo {a}"
            bn = "bờ sông" if b == -2 else "bể" if b == -1 else f"gáo {b}"

            if a == -2:
                state[b] = inp[b]
            elif b == -1:
                current += state[a]
                state[a] = 0
            else:
                dif = min([state[a], inp[b] - state[b]])
                state[a] -= dif
                state[b] += dif

            print(f"{an} -> {bn}, bể: {current}, state: {state}")
