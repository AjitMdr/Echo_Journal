import 'package:echo_journal_1/core/configs/theme/gradient-bg-pattern.dart';
import 'package:echo_journal_1/core/configs/theme/theme-provider.dart';
import 'package:echo_journal_1/features/authentication/pages/forgot_password_page.dart';
import 'package:echo_journal_1/features/authentication/pages/signup_page.dart';
import 'package:echo_journal_1/features/authentication/widgets/login_form.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

class LoginPage extends StatelessWidget {
  const LoginPage({super.key});

  @override
  Widget build(BuildContext context) {
    final themeProvider = Provider.of<ThemeProvider>(context);
    final isDarkMode = themeProvider.isDarkMode;

    return Scaffold(
      body: GradientBackgroundWithPattern(
        child: SafeArea(
          child: Center(
            child: SingleChildScrollView(
              padding: const EdgeInsets.symmetric(horizontal: 24),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  // Container for the login form and "Don't have an account?" button
                  Container(
                    padding: const EdgeInsets.all(24),
                    decoration: BoxDecoration(
                      color: isDarkMode ? Colors.grey[850] : Colors.white,
                      borderRadius: BorderRadius.circular(30),
                      boxShadow: [
                        BoxShadow(
                          color: isDarkMode
                              ? Colors.black.withOpacity(0.4)
                              : Colors.black.withOpacity(0.1),
                          blurRadius: 10,
                          offset: const Offset(0, 5),
                        ),
                      ],
                    ),
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        const LoginForm(), // Reusable LoginForm widget
                        const SizedBox(height: 16),
                        // Forgot Password Button
                        TextButton(
                          onPressed: () => Navigator.push(
                            context,
                            MaterialPageRoute(
                              builder: (context) => const ForgotPasswordPage(),
                            ),
                          ),
                          child: Text(
                            "Forgot Password?",
                            style: TextStyle(
                              color: isDarkMode
                                  ? Color.fromARGB(255, 174, 170, 171)
                                  : Color(0xFFFF758C),
                            ),
                          ),
                        ),
                        // "Don't have an account?" TextButton inside the container
                        TextButton(
                          onPressed: () => Navigator.push(
                            context,
                            MaterialPageRoute(
                              builder: (context) => const RegisterPageUI(),
                            ),
                          ),
                          child: Text(
                            "Don't have an account? Create Account",
                            style: TextStyle(
                              color: isDarkMode
                                  ? Color.fromARGB(255, 174, 170, 171)
                                  : Color(0xFFFF758C),
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}
