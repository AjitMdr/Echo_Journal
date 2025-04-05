import 'package:echo_fe/features/authentication/pages/forgot_password_page.dart';
import 'package:echo_fe/features/authentication/pages/signup_page.dart';
import 'package:echo_fe/features/authentication/widgets/login_form.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../../../core/configs/theme/gradient-bg-pattern.dart';
import '../../../core/configs/theme/theme-provider.dart';
import '../../../services/auth/login_service.dart';
import '../../home/home_page.dart';
import '../../../core/providers/subscription_provider.dart';
import '../../../features/authentication/pages/otp_verification_page.dart';

class LoginPage extends StatefulWidget {
  const LoginPage({super.key});

  @override
  _LoginPageState createState() => _LoginPageState();
}

class _LoginPageState extends State<LoginPage> {
  final _formKey = GlobalKey<FormState>();
  final _emailController = TextEditingController();
  final _passwordController = TextEditingController();
  bool _isLoading = false;
  String? _error;

  Future<void> _handleLogin() async {
    if (!_formKey.currentState!.validate()) return;

    setState(() {
      _isLoading = true;
      _error = null;
    });

    try {
      final email = _emailController.text.trim();
      final password = _passwordController.text;

      final response = await AuthService.login(email, password);

      if (mounted) {
        if (response['success'] == true) {
          // Check subscription status after successful login
          final subscriptionProvider =
              Provider.of<SubscriptionProvider>(context, listen: false);
          await subscriptionProvider.checkSubscription();

          // Navigate to home and remove all previous routes
          Navigator.of(context).pushNamedAndRemoveUntil(
            '/',
            (Route<dynamic> route) => false,
          );
        } else {
          // Handle unverified account
          if (response['needs_verification'] == true) {
            // Navigate to OTP verification page
            Navigator.pushReplacement(
              context,
              MaterialPageRoute(
                builder: (context) => OTPVerificationPage(
                  email: response['email'] ?? email,
                ),
              ),
            );
          } else {
            setState(() {
              _error = response['error'];
              _isLoading = false;
            });
          }
        }
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _error = e.toString();
          _isLoading = false;
        });
      }
    }
  }

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

  @override
  void dispose() {
    _emailController.dispose();
    _passwordController.dispose();
    super.dispose();
  }
}
