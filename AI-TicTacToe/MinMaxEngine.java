package com.example.tictactoe;

import java.util.ArrayList;
import java.util.List;

public class MinMaxEngine {
    private int[][] board;
    private int currentPlayer;
    private boolean ended;
    public Point computerMove;


    public static final int NO_PLAYER = 0;
    public static final int PLAYER_X = 1;
    public static final int PLAYER_0 = 2;

    public MinMaxEngine() {
        board = new int[3][3];
        newGame();
    }

    public boolean isEnded() {
        return ended;
    }

    public int play(int x, int y) {
        if (!ended  &&  board[x][y] == NO_PLAYER) {
            board[x][y] = currentPlayer;
            changePlayer();
        }
        return checkEnd();
    }

    public void changePlayer() {
        currentPlayer = (currentPlayer == PLAYER_X ? PLAYER_0 : PLAYER_X);
    }

    public int getElt(int x, int y) {
        return board[x][y];
    }

    public void newGame() {
        for (int i = 0; i  < 3; i++) {
            for(int j = 0; j < 3; j ++){
                board[i][j] = NO_PLAYER;
            }
        }

        currentPlayer = PLAYER_0;
        ended = false;
    }

    public int checkEnd() {
        for(int i = 0; i < 3; i ++){
            //checking horizontally
            if(board[i][0]!=NO_PLAYER && board[i][0] == board[i][1] && board[i][0] == board[i][2]){
                ended = true;
                return board[i][0];
            }
            //checking vertically
            if(board[0][i]!=NO_PLAYER && board[0][i] == board[1][i] && board[0][i] == board[2][i]){
                ended = true;
                return board[0][i];
            }
        }

        //checking diagonally
        if(board[0][0]!=NO_PLAYER && board[0][0]==board[1][1] && board[1][1]==board[2][2]){
            ended = true;
            return board[0][0];
        }
        if(board[2][0]!=NO_PLAYER && board[2][0]==board[1][1] && board[1][1]==board[0][2]){
            ended = true;
            return board[2][0];
        }

        //checking for other cells
        for(int i = 0; i < 3; i ++){
            for(int j = 0; j < 3; j ++){
                if(board[i][j] == NO_PLAYER)
                    return NO_PLAYER;
            }
        }

        return -1;
    }

    public List<Point> getAvailableCells(){
        List<Point> availableCells = new ArrayList<>();

        for(int i = 0; i < 3; i ++){
            for(int j = 0; j < 3; j ++){
                if(board[i][j]==NO_PLAYER){
                    availableCells.add(new Point(i,j));
                }
            }
        }
        return availableCells;
    }

    public boolean placeAMove(Point point, int player){
        if(board[point.x][point.y] != NO_PLAYER){
            return false;
        }
        board[point.x][point.y] = player;
        return true;
    }

    public int computer(int x, int y) {

        Point userMove = new Point(x, y);
        placeAMove(userMove, PLAYER_0);

        minmax(0, PLAYER_X);
        System.out.println("Computer choose position: " + computerMove);

        placeAMove(computerMove, PLAYER_X);
        changePlayer();
        //displayBoard();

        if(hasPlayerWon(PLAYER_0))
            return PLAYER_0;
        else if(hasPlayerWon(PLAYER_X))
            return PLAYER_X;
        else
            return NO_PLAYER;
    }

    public boolean hasPlayerWon(int player){
        if((board[0][0]==board[1][1] && board[0][0]==board[2][2] && board[0][0]==player)
                || (board[0][2]==board[1][1] && board[0][2]==board[2][0] && board[0][2]==player)){
            return true;
        }

        for(int i = 0; i < 3; i++){
            if((board[i][0]==board[i][1]&&board[i][0]==board[i][2]&& board[i][0]==player)
                    || (board[0][i]==board[1][i] && board[0][i]==board[2][i] && board[0][i]==player)){
                return true;
            }
        }
        return false;
    }

    public int minmax(int depth, int turn){
        //user is 0 and computer is X
        if(hasPlayerWon(PLAYER_X))
            return 1;
        if(hasPlayerWon(PLAYER_0))
            return -1;

        List<Point> availableCells = getAvailableCells();

        if(availableCells.isEmpty())
            return 0;

        int min = Integer.MAX_VALUE;
        int max = Integer.MIN_VALUE;

        //want to maximize the value for 0 and minimize it for X
        for(int i = 0; i < availableCells.size(); i++){
            Point point = availableCells.get(i);

            if(turn == PLAYER_X){
                placeAMove(point, PLAYER_X);
                int currentScore = minmax(depth+1, PLAYER_0);
                max = Math.max(currentScore, max);

                if(depth ==0){
                    System.out.println("computer score for position " + point + " = " + currentScore);
                }

                if(currentScore >= 0)
                    if(depth == 0)
                        computerMove = point;

                if(currentScore == 1){
                    board[point.x][point.y] = NO_PLAYER;
                    break;
                }

                if(i == availableCells.size() && max < 0){
                    if(depth == 0)
                        computerMove = point;
                }
            }
            else if(turn == PLAYER_0){
                placeAMove(point, PLAYER_0);
                int currentScore = minmax(depth+1, PLAYER_X);
                min = Math.min(currentScore, min);

                if(min == -1){
                    board[point.x][point.y] = NO_PLAYER;
                    break;
                }
            }
            board[point.x][point.y] = NO_PLAYER;
        }
        return turn == PLAYER_X ? max : min;
    }
}
