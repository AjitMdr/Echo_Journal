import 'package:echo_journal1/common/widgets/error_message.dart';
import 'package:echo_journal1/common/widgets/success_message.dart';
import 'package:echo_journal1/core/configs/theme/app-styles.dart';
import 'package:echo_journal1/core/configs/theme/gradient-bg-pattern.dart';
import 'package:echo_journal1/core/configs/theme/theme-provider.dart';
import 'package:echo_journal1/features/authentication/pages/reset_password_page.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:echo_journal1/services/auth/forgot_password_service.dart';
import 'login_page.dart';

class ForgotPasswordPage extends StatefulWidget {
  const ForgotPasswordPage({super.key});

  @override
  _ForgotPasswordPageState createState() => _ForgotPasswordPageState();
}

class _ForgotPasswordPageState extends State<ForgotPasswordPage> {
  final TextEditingController _emailController = TextEditingController();
  bool _isLoading = false;
  String _errorMessage = '';
  String _successMessage = '';

  @override
  void dispose() {
    _emailController.dispose();
    super.dispose();
  }

  Future<void> _sendOTP() async {
    final String email = _emailController.text.trim();
    if (email.isEmpty) {
      setState(() {
        _errorMessage = 'Please enter your email';
        _successMessage = '';
      });
      return;
    }

    // Add email validation
    final bool emailValid = RegExp(r'^[^@]+@[^@]+\.[^@]+').hasMatch(email);
    if (!emailValid) {
      setState(() {
        _errorMessage = 'Please enter a valid email address';
        _successMessage = '';
      });
      return;
    }

    setState(() {
      _isLoading = true;
      _errorMessage = '';
      _successMessage = '';
    });

    try {
      final result = await ForgotPasswordService.requestPasswordResetOTP(email);

      if (mounted) {
        if (result['success'] == true) {
          setState(() {
            _successMessage =
                result['message'] ?? 'OTP sent successfully! Check your email.';
          });

          // Wait a moment to show the success message before navigating
          await Future.delayed(const Duration(seconds: 1));

          if (mounted) {
            Navigator.push(
              context,
              MaterialPageRoute(
                builder: (context) => ResetPasswordPage(email: email),
              ),
            );
          }
        } else {
          setState(() {
            _errorMessage = result['error'] ?? 'Failed to send OTP';
          });
        }
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _errorMessage = 'An error occurred. Please try again.';
          _successMessage = '';
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
                        Text(
                          'Forgot Password',
                          style: TextStyle(
                            fontSize: 32,
                            fontWeight: FontWeight.bold,
                            color: isDarkMode
                                ? const Color.fromARGB(255, 255, 255, 255)
                                : const Color(0xFFFF758C),
                          ),
                        ),
                        const SizedBox(height: 8),
                        Text(
                          'Enter your email to receive an OTP',
                          style: TextStyle(
                            fontSize: 16,
                            color: isDarkMode
                                ? Colors.grey[300]
                                : Colors.grey[700],
                          ),
                        ),
                        const SizedBox(height: 24),
                        TextFormField(
                          controller: _emailController,
                          keyboardType: TextInputType.emailAddress,
                          style: TextStyle(
                            color: isDarkMode ? Colors.white : Colors.black,
                          ),
                          decoration: AppStyles.inputDecoration.copyWith(
                            labelText: 'Email',
                            hintText: 'Enter your email address',
                            labelStyle: TextStyle(
                              color: isDarkMode ? Colors.white : Colors.black,
                            ),
                            hintStyle: TextStyle(
                              color: isDarkMode
                                  ? Colors.white.withOpacity(0.7)
                                  : Colors.black.withOpacity(0.7),
                            ),
                            fillColor: isDarkMode
                                ? Colors.grey[800]
                                : Colors.grey[100],
                            filled: true,
                            prefixIcon: Icon(
                              Icons.email_outlined,
                              color: isDarkMode ? Colors.white : Colors.black,
                            ),
                            border: OutlineInputBorder(
                              borderRadius: BorderRadius.circular(15),
                              borderSide: BorderSide(
                                color: isDarkMode
                                    ? Colors.grey[600]!
                                    : Colors.grey[300]!,
                              ),
                            ),
                          ),
                        ),
                        const SizedBox(height: 20),
                        if (_errorMessage.isNotEmpty)
                          ErrorMessageBox(errorMessage: _errorMessage),
                        if (_successMessage.isNotEmpty)
                          SuccessMessageBox(successMessage: _successMessage),
                        const SizedBox(height: 24),
                        SizedBox(
                          width: double.infinity,
                          height: 50,
                          child: _isLoading
                              ? Center(
                                  child: CircularProgressIndicator(
                                    color: isDarkMode
                                        ? Colors.white
                                        : Colors.black,
                                  ),
                                )
                              : ElevatedButton(
                                  onPressed: _sendOTP,
                                  style: ElevatedButton.styleFrom(
                                    backgroundColor: isDarkMode
                                        ? Colors.grey[800]
                                        : const Color(0xFFFF758C),
                                    elevation: 3,
                                    shape: RoundedRectangleBorder(
                                      borderRadius: BorderRadius.circular(15),
                                    ),
                                  ),
                                  child: const Text(
                                    'Send OTP',
                                    style: TextStyle(
                                      fontSize: 16,
                                      fontWeight: FontWeight.bold,
                                    ),
                                  ),
                                ),
                        ),
                        const SizedBox(height: 16),
                        TextButton(
                          onPressed: () => Navigator.pushReplacement(
                            context,
                            MaterialPageRoute(
                              builder: (context) => const LoginPage(),
                            ),
                          ),
                          child: Text(
                            "Remembered your password? Login",
                            style: TextStyle(
                              color: isDarkMode
                                  ? const Color.fromARGB(255, 174, 170, 171)
                                  : const Color(0xFFFF758C),
                              fontWeight: FontWeight.bold,
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
