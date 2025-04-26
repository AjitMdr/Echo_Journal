import 'package:dio/dio.dart';
import 'package:echo_journal1/core/configs/api_config.dart';

class BaseService {
  static final Dio dio = Dio(BaseOptions(
    baseUrl: ApiConfig.baseUrl,
    headers: {
      'Content-Type': 'application/json',
    },
    validateStatus: (status) {
      return status != null && status < 500;
    },
  ));
}
