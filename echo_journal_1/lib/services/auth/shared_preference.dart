import 'package:shared_preferences/shared_preferences.dart';

/// 🔧 Utility class for handling shared preferences operations
class SharedPreferenceService {
  static Future<void> printSharedPreferences() async {
    final prefs = await SharedPreferences.getInstance();
    final keys = prefs.getKeys();
    print('🗄️ Shared Preferences Storage:');
    for (String key in keys) {
      print('🔑 $key: ${prefs.get(key)}');
    }
  }
}
