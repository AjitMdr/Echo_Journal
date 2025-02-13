// terms_of_use_page.dart
import 'package:flutter/material.dart';

class TermsOfUsePage extends StatelessWidget {
  const TermsOfUsePage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Terms of Use')),
      body: const Center(
        child: Text('Here are the Terms of Use...'),
      ),
    );
  }
}


