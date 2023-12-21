import heapq


class Node:
    def __init__(self, label, goal_cost):
        self.label = label
        self.cost = 10000
        self.goal_cost = goal_cost
        self.save_cost = None
        self.pr = []
        self.chld = []

    def __repr__(self):
        return str(
            dict({"label": self.label, "cost": self.cost, "goal cost": self.goal_cost})
        )

    def __eq__(self, other):
        return self.label == other.label

    def __lt__(self, other):
        if self.save_cost == 10000:
            return self.goal_cost + self.cost < other.goal_cost + other.cost
        else:
            return self.cost < other.cost

    def get_label(self):
        return self.label

    def neighbors(self):
        return self.chld + self.pr


class Tree:
    def __init__(self):
        self.nodes = []
        self.edges = []

    def add_nodes(self, data):
        for node in data:
            self.nodes.append(node)

    def get_index(self, node):
        for i, n in enumerate(self.nodes):
            if n.get_label() == node.get_label():
                return i
        return -1

    def add_edges(self, tuple_edges):
        for t in tuple_edges:
            start_label = t[0]
            end_label = t[1]
            w = t[2]

            index_start_label = self.get_index(Node(start_label, None))
            index_end_label = self.get_index(Node(end_label, None))

            self.nodes[index_start_label].chld.append(self.nodes[index_end_label])
            self.nodes[index_end_label].pr.append(self.nodes[index_start_label])
            self.edges.append(
                (self.nodes[index_start_label], self.nodes[index_end_label], t[2])
            )

    def show_nodes(self):
        return [node.get_data() for node in self.nodes]

    def get_edge(self, start_node, end_node):
        try:
            return [
                edges
                for edges in self.edges
                if edges[0] == start_node and edges[1] == end_node
            ][0]
        except:
            return None


def update_cost(tree, current_node, prev_node):
    if tree.get_edge(prev_node, current_node) is not None:
        if (
            current_node.cost
            > prev_node.cost + tree.get_edge(prev_node, current_node)[2]
        ):
            current_node.cost = (
                prev_node.cost + tree.get_edge(prev_node, current_node)[2]
            )


def A_Star(tree, start, end):
    frontier = [start]
    heapq.heapify(frontier)
    explored = []

    while len(frontier) > 0:
        state = heapq.heappop(frontier)
        explored.append(state)
        print(state)
        if state == end:
            return explored
        for child in state.neighbors():
            update_cost(tree, child, state)
            if child.get_label() not in list(
                set(node.get_label() for node in frontier + explored)
            ):
                heapq.heappush(frontier, child)
    return False


if __name__ == "__main__":
    tree = Tree()
    tree.add_nodes(
        [
            Node("A", 6),
            Node("B", 3),
            Node("C", 4),
            Node("D", 5),
            Node("E", 3),
            Node("F", 1),
            Node("G", 6),
            Node("H", 2),
            Node("I", 5),
            Node("J", 4),
            Node("K", 2),
            Node("L", 0),
            Node("M", 4),
            Node("N", 0),
            Node("O", 4),
        ]
    )

    tree.add_edges(
        [
            ("A", "B", 2),
            ("A", "C", 1),
            ("A", "D", 3),
            ("B", "E", 5),
            ("B", "F", 4),
            ("C", "G", 6),
            ("C", "H", 3),
            ("D", "I", 2),
            ("D", "J", 4),
            ("F", "K", 2),
            ("F", "L", 1),
            ("F", "M", 4),
            ("H", "N", 2),
            ("H", "O", 4),
        ]
    )
    tree.nodes[0].cost = 0
    result = A_Star(tree, tree.nodes[0], tree.nodes[14])
    if result:
        s = "explored: "
        for i in result:
            s += i.label + " "
            print(s)
    else:
        print("404 Not Found!")
