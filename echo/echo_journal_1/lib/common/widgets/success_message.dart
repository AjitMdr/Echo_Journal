import 'package:flutter/material.dart';
import 'dart:async';

class SuccessMessageBox extends StatefulWidget {
  final String successMessage;

  const SuccessMessageBox({super.key, required this.successMessage});

  @override
  _SuccessMessageBoxState createState() => _SuccessMessageBoxState();
}

class _SuccessMessageBoxState extends State<SuccessMessageBox> {
  bool _isVisible = true;

  @override
  void initState() {
    super.initState();
    // Hide the success message after 2 seconds
    Timer(const Duration(seconds: 5), () {
      setState(() {
        _isVisible = false;
      });
    });
  }

  @override
  Widget build(BuildContext context) {
    return _isVisible
        ? Container(
          margin: const EdgeInsets.only(bottom: 16),
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            color: Colors.green.shade100,
            borderRadius: BorderRadius.circular(8),
            border: Border.all(color: Colors.green),
          ),
          child: Row(
            children: [
              Icon(Icons.check_circle_outline, color: Colors.green),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  widget.successMessage,
                  style: const TextStyle(color: Colors.green, fontSize: 16),
                ),
              ),
            ],
          ),
        )
        : SizedBox.shrink(); // Return an empty widget if not visible
  }
}
