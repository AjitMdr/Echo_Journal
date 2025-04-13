import 'package:echo_journal1/constants/esewa_const.dart';
import 'package:esewa_flutter_sdk/esewa_flutter_sdk.dart';
import 'package:esewa_flutter_sdk/esewa_config.dart';
import 'package:esewa_flutter_sdk/esewa_payment.dart';
import 'package:esewa_flutter_sdk/esewa_payment_success_result.dart';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

class Esewa {
  Future<void> pay({
    required String productId,
    required String productName,
    required String productPrice,
    required Function(EsewaPaymentSuccessResult) onSuccess,
    required Function(String) onFailure,
    required Function() onCancelled,
  }) async {
    try {
      EsewaFlutterSdk.initPayment(
        esewaConfig: EsewaConfig(
          environment: Environment.test,
          clientId: kEsewaClientId,
          secretId: kEsewaSecretKey,
        ),
        esewaPayment: EsewaPayment(
          productId: productId,
          productName: productName,
          productPrice: productPrice,
          callbackUrl: 'https://example.com/callback',
        ),
        onPaymentSuccess: (EsewaPaymentSuccessResult data) async {
          debugPrint("Towards verification");
          final verificationResult = await verifyTransaction(data);

          if (verificationResult) {
            onSuccess(data);
          } else {
            onFailure('Payment verification failed');
          }
        },
        onPaymentFailure: (data) {
          debugPrint(":::FAILURE::: => $data");
          onFailure(data.toString());
        },
        onPaymentCancellation: (data) {
          debugPrint(":::CANCELLATION::: => $data");
          onCancelled();
        },
      );
    } catch (e) {
      debugPrint('eSewa payment error: $e');
      onFailure(e.toString());
    }
  }

  Future<bool> verifyTransaction(EsewaPaymentSuccessResult result) async {
    try {
      // Using refId method for verification
      final response = await http.get(
        Uri.parse(
            'https://rc.esewa.com.np/mobile/transaction?txnRefId=${result.refId}'),
        headers: {
          'merchantId': kEsewaClientId,
          'merchantSecret': kEsewaSecretKey,
          'Content-Type': 'application/json',
        },
      );

      if (response.statusCode == 200) {
        final List<dynamic> data = json.decode(response.body);
        debugPrint(":::DATA::: => $data");
        if (data.isNotEmpty) {
          final transactionStatus = data[0]['transactionDetails']['status'];
          return transactionStatus == 'COMPLETE';
        }
      }
      return false;
    } catch (e) {
      debugPrint('Transaction verification error: $e');
      return false;
    }
  }
}
