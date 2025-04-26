import 'package:echo_journal1/features/authentication/pages/forgot_password_page.dart';
import 'package:echo_journal1/features/authentication/pages/signup_page.dart';
import 'package:echo_journal1/features/authentication/widgets/login_form.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../../../core/configs/theme/gradient-bg-pattern.dart';
import '../../../core/configs/theme/theme-provider.dart';
import '../../../services/auth/login_service.dart';
import 'verify_2fa_login_page.dart';
import 'verify_account_page.dart';
import '../../../features/widgets/navbar.dart';

class LoginPage extends StatefulWidget {
  const LoginPage({Key? key}) : super(key: key);

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
    if (_formKey.currentState!.validate()) {
      setState(() {
        _isLoading = true;
        _error = null;
      });

      try {
        final response = await AuthService.login(
          _emailController.text,
          _passwordController.text,
        );

        if (mounted) {
          if (response['requires_2fa'] == true) {
            // Navigate to 2FA verification page
            Navigator.push(
              context,
              MaterialPageRoute(
                builder: (context) => Verify2FALoginPage(
                  email: response['email'],
                ),
              ),
            );
          } else if (response['success']) {
            // Normal login success
            Navigator.pushAndRemoveUntil(
              context,
              MaterialPageRoute(builder: (context) => const NavBar()),
              (route) => false,
            );
          } else if (response['needs_verification']) {
            // Handle account verification
            Navigator.push(
              context,
              MaterialPageRoute(
                builder: (context) => VerifyAccountPage(
                  email: response['email'],
                ),
              ),
            );
          } else {
            setState(() {
              _error = response['error'];
            });
          }
        }
      } catch (e) {
        if (mounted) {
          setState(() {
            _error = e.toString();
          });
        }
      } finally {
        if (mounted) {
          setState(() {
            _isLoading = false;
          });
        }
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
                  // Logo
                  Padding(
                    padding: const EdgeInsets.only(bottom: 32.0),
                    child: Image.asset(
                      'assets/images/logo.png',
                      height: 120,
                      width: 120,
                    ),
                  ),
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
