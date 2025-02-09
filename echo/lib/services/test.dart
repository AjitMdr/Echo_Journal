import 'package:http/http.dart' as http;
import '../config.dart';

class ConnectionTest {
  static Future<void> testWithHttp() async {
    try {
      print('\n=== Testing with basic HTTP ===');
      print('URL: ${Config.baseUrl}');
      
      final response = await http.get(
        Uri.parse('${Config.baseUrl}/'),
      ).timeout(
        Duration(seconds: 10),
        onTimeout: () {
          print('Request timed out');
          throw Exception('Timeout');
        },
      );
      
      print('Connection successful!');
      print('Status code: ${response.statusCode}');
      print('Body: ${response.body}');
    } catch (e) {
      print('Error: $e');
      print('\nTroubleshooting steps:');
      print('1. Check if Android Emulator is running');
      print('2. Verify server is running on port 8000');
      print('3. Try accessing server from host machine: http://localhost:8000');
      print('4. Check server logs for incoming requests');
    }
  }
}