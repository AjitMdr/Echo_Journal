import 'package:flutter/material.dart';
import 'dart:async';

class ErrorMessageBox extends StatefulWidget {
  final String errorMessage;

  const ErrorMessageBox({super.key, required this.errorMessage});

  @override
  _ErrorMessageBoxState createState() => _ErrorMessageBoxState();
}

class _ErrorMessageBoxState extends State<ErrorMessageBox> {
  bool _isVisible = true;
  Timer? _timer; // Store timer reference

  @override
  void initState() {
    super.initState();
    // Hide the error message after 2 seconds
    _timer = Timer(const Duration(seconds: 2), () {
      if (mounted) {
        // Check if the widget is still in the tree
        setState(() {
          _isVisible = false;
        });
      }
    });
  }

  @override
  void dispose() {
    _timer?.cancel(); // Cancel the timer when the widget is disposed
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return _isVisible
        ? Container(
            padding: const EdgeInsets.all(12),
            margin: const EdgeInsets.only(bottom: 10),
            decoration: BoxDecoration(
              color: Colors.red.withOpacity(0.1),
              borderRadius: BorderRadius.circular(10),
              border: Border.all(color: Colors.red),
            ),
            child: Row(
              children: [
                const Icon(Icons.error_outline, color: Colors.red),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(
                    widget.errorMessage,
                    style: const TextStyle(color: Colors.red),
                  ),
                ),
              ],
            ),
          )
        : const SizedBox.shrink(); // Return an empty widget if not visible
  }
}
