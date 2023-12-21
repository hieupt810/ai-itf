from TreeNode import Tree


def MaxValue(node, alpha, beta):
    if len(node.children) == 0:
        return node
    node.value = -1 * 10**5
    for child in node.children:
        temp = MinValue(child, alpha, beta)
        if temp.value > node.value:
            node.value = temp.value
        if child.value >= beta:
            return child
        if child.value > alpha:
            alpha = child.value
    return node


def MinValue(node, alpha, beta):
    if len(node.children) == 0:
        return node
    node.value = 10**5
    for child in node.children:
        temp = MaxValue(child, alpha, beta)
        if temp.value < node.value:
            node.value = temp.value
        if child.value <= alpha:
            return child
        if child.value < beta:
            beta = child.value
    return node


def Alpha_Beta_Search(state):
    MaxValue(state, -100000, 100000)


if __name__ == "__main__":
    A = Tree("A")
    B = Tree("B")
    C = Tree("C")
    D = Tree("D")
    E = Tree("E")
    F = Tree("F")
    G = Tree("G")
    H = Tree("H")
    I = Tree("I")
    J = Tree("J")
    K = Tree("K")
    L = Tree("L")
    M = Tree("M")
    N = Tree("N")
    Z = Tree("Z")
    A.add_child(B)
    A.add_child(C)
    B.add_child(D)
    B.add_child(E)
    C.add_child(F)
    C.add_child(G)
    D.add_child(H)
    D.add_child(I)
    E.add_child(J)
    E.add_child(K)
    F.add_child(M)
    F.add_child(N)
    G.add_child(L)
    G.add_child(Z)
    H.value = 2
    I.value = 9
    J.value = 7
    K.value = 4
    M.value = 8
    N.value = 9
    L.value = 3
    Z.value = 5
    Alpha_Beta_Search(A)
    print(A.value)
