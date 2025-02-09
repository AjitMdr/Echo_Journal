import 'package:flutter/material.dart';

class HomePage extends StatelessWidget {
  final String username; // Add a field to accept the username

  // Constructor to accept the username
  const HomePage({Key? key, required this.username}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Home Page')),
      body: Center(
        child: Text(
          'Welcome, $username!', // Display the username here
          style: const TextStyle(fontSize: 20),
        ),
      ),
    );
  }
}
