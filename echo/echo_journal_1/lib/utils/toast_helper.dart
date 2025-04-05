import 'package:flutter/material.dart';
import 'package:echo_fe/widgets/toast_notification.dart';

class ToastHelper {
  static void showSuccess(BuildContext context, String message) {
    ToastNotification.show(context, message: message, type: ToastType.success);
  }

  static void showError(BuildContext context, String message) {
    ToastNotification.show(context, message: message, type: ToastType.error);
  }

  static void showInfo(BuildContext context, String message) {
    ToastNotification.show(context, message: message, type: ToastType.info);
  }

  static void showWarning(BuildContext context, String message) {
    ToastNotification.show(context, message: message, type: ToastType.warning);
  }
}
