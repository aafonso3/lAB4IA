import java.io.*;

public class NetworkUtils {


    public static void saveNetwork(Network network, String filePath) throws IOException {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(filePath))) {
            oos.writeObject(network);
        }
    }

 
    public static Network loadNetwork(String filePath) throws IOException, ClassNotFoundException {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filePath))) {
            return (Network) ois.readObject();
        }
    }
}
