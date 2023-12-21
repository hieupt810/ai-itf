def BFS(initialState, goal):
    frontier = [initialState]
    explored = []
    while frontier:
        state = frontier.pop(0)
        explored.append(state)
        if goal == state:
            return explored
        for neighbor in sorted(graph[state]):
            if (neighbor not in explored) and (neighbor not in frontier):
                frontier.append(neighbor)
    return False


if __name__ == "__main__":
    graph = {
        "S": ["A", "B", "C"],
        "A": ["B", "D", "S"],
        "B": ["A", "C", "D", "F", "G", "S"],
        "C": ["B", "F", "S"],
        "D": ["A", "B", "E"],
        "E": ["D", "F", "G"],
        "F": ["B", "C", "E", "H"],
        "G": ["B", "E", "H"],
        "H": ["F", "G"],
    }
    res = BFS("S", "G")
    if res:
        s = "Explored: "
        for i in res:
            s += i + " "
            print(s)
    else:
        print("Khong tim thay duong di")
