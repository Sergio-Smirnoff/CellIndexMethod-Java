public class Particle {
    private int id;
    private double x,y;
    private int cellX, cellY;

    public Particle(int id, double x, double y){
        this.id = id;
    }

    public int getId() {
        return id;
    }
    public double getX() {
        return x;
    }
    public double getY() {
        return y;
    }
    public int getCellX() {
        return cellX;
    }
    public int getCellY() {
        return cellY;
    }
    public void setCell(int cellX, int cellY) {
        this.cellX = cellX;
        this.cellY = cellY;
    }

    public void setPosition(double x, double y) {
        this.x = x;
        this.y = y;
        // Update cell indices based on new position
    }
}