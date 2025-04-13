import 'package:flutter/material.dart';

enum ToastType { success, error, info, warning }

class ToastNotification extends StatelessWidget {
  final String message;
  final ToastType type;

  const ToastNotification({Key? key, required this.message, required this.type})
    : super(key: key);

  static void show(
    BuildContext context, {
    required String message,
    required ToastType type,
  }) {
    OverlayState? overlay = Overlay.of(context);

    OverlayEntry overlayEntry = OverlayEntry(
      builder:
          (context) => Positioned(
            bottom: MediaQuery.of(context).size.height * 0.1,
            width: MediaQuery.of(context).size.width,
            child: Material(
              color: Colors.transparent,
              child: SafeArea(
                child: Center(
                  child: Container(
                    margin: const EdgeInsets.symmetric(horizontal: 20),
                    padding: const EdgeInsets.symmetric(
                      horizontal: 24,
                      vertical: 12,
                    ),
                    decoration: BoxDecoration(
                      color: _getBackgroundColor(type),
                      borderRadius: BorderRadius.circular(8),
                      boxShadow: [
                        BoxShadow(
                          color: Colors.black.withOpacity(0.15),
                          offset: const Offset(0, 4),
                          blurRadius: 8,
                        ),
                      ],
                    ),
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        _getIcon(type),
                        const SizedBox(width: 12),
                        Flexible(
                          child: Text(
                            message,
                            style: const TextStyle(
                              color: Colors.white,
                              fontSize: 14,
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            ),
          ),
    );

    overlay.insert(overlayEntry);

    Future.delayed(const Duration(seconds: 3), () {
      overlayEntry.remove();
    });
  }

  static Color _getBackgroundColor(ToastType type) {
    switch (type) {
      case ToastType.success:
        return Colors.green;
      case ToastType.error:
        return Colors.red;
      case ToastType.info:
        return Colors.blue;
      case ToastType.warning:
        return Colors.orange;
    }
  }

  static Widget _getIcon(ToastType type) {
    IconData iconData;
    switch (type) {
      case ToastType.success:
        iconData = Icons.check_circle;
        break;
      case ToastType.error:
        iconData = Icons.error;
        break;
      case ToastType.info:
        iconData = Icons.info;
        break;
      case ToastType.warning:
        iconData = Icons.warning;
        break;
    }
    return Icon(iconData, color: Colors.white);
  }

  @override
  Widget build(BuildContext context) {
    return Container(); // This widget is only used for its static methods
  }
}
