class TicTacToe:
    # Khởi tạo trạng thái trò chơi
    def __init__(self, initialState) -> None:
        self.gameState = initialState
        self.bestNextMove = []
        self.moveToWin = 10**2

    # Tìm người chơi sẽ được đánh kế tiếp vào trạng thái trò chơi
    def nextPlayer(self) -> int:
        count = [0 for _ in range(3)]
        for i in range(3):
            for j in range(3):
                count[self.gameState[i][j]] += 1

        if count[1] == count[2]:
            return 1
        elif count[1] > count[2]:
            return 2
        else:
            return -1

    # Tìm người chiến thắng của trò chơi
    def winner(self) -> int:
        for i in range(3):
            if (
                self.gameState[i][0] == self.gameState[i][1]
                and self.gameState[i][1] == self.gameState[i][2]
            ) or (
                self.gameState[0][i] == self.gameState[1][i]
                and self.gameState[1][i] == self.gameState[2][i]
            ):
                return self.gameState[i][1]

        if (
            self.gameState[0][0] == self.gameState[1][1]
            and self.gameState[1][1] == self.gameState[2][2]
        ) or (
            self.gameState[0][2] == self.gameState[1][1]
            and self.gameState[1][1] == self.gameState[2][0]
        ):
            return self.gameState[1][1]

        return -1

    # In trạng thái trò chơi
    def printState(self):
        for i in range(3):
            for j in range(3):
                if self.gameState[i][j] == 0:
                    print("_", end=" ")
                elif self.gameState[i][j] == 1:
                    print("O", end=" ")
                else:
                    print("X", end=" ")
            print()
        print()

    # Sinh trạng thái trò chơi
    def generateGameState(self, i=0, j=0, after_step=0, print: bool = False):
        if print:
            self.printState()

        # Minimax for X win
        if self.winner() == 2:
            self.moveToWin = min(self.moveToWin, after_step)
            return 2

        if self.winner() != -1:
            return 0

        for i in range(3):
            for j in range(3):
                if self.gameState[i][j] == 0:
                    self.gameState[i][j] = self.nextPlayer()
                    game = self.generateGameState(after_step + 1, print=False)
                    if game:
                        self.bestNextMove = [i + 1, j + 1]
                    self.gameState[i][j] = 0


if __name__ == "__main__":
    game = TicTacToe([[1, 0, 0], [2, 2, 1], [0, 1, 0]])
    game.generateGameState(print=False)
    # In vị trí hàng, cột của nước đi tốt nhất tiếp theo
    print(game.bestNextMove)
