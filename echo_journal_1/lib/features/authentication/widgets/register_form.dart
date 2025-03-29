import 'package:echo_journal_1/common/widgets/error_message.dart';
import 'package:echo_journal_1/core/configs/theme/app-styles.dart';
import 'package:echo_journal_1/core/configs/theme/theme-provider.dart';
import 'package:echo_journal_1/features/authentication/pages/otp_verification_page.dart';
import 'package:echo_journal_1/services/auth/register_service.dart';
import 'package:echo_journal_1/utils/toast_helper.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

class RegisterForm extends StatefulWidget {
  const RegisterForm({super.key});

  @override
  _RegisterFormState createState() => _RegisterFormState();
}

class _RegisterFormState extends State<RegisterForm> {
  final _formKey = GlobalKey<FormState>();
  final _emailController = TextEditingController();
  final _usernameController = TextEditingController();
  final _passwordController = TextEditingController();
  final _confirmPasswordController = TextEditingController();
  bool _obscurePassword = true;
  bool _obscureConfirmPassword = true;
  bool _isLoading = false;
  String _errorMessage = '';

  @override
  void dispose() {
    _emailController.dispose();
    _usernameController.dispose();
    _passwordController.dispose();
    _confirmPasswordController.dispose();
    super.dispose();
  }

  Future<void> _handleSignup() async {
    if (_formKey.currentState!.validate()) {
      setState(() {
        _isLoading = true;
        _errorMessage = ''; // Clear previous error message
      });

      try {
        final Map<String, dynamic> response = await RegisterService.register(
          _usernameController.text,
          _passwordController.text,
          _emailController.text,
          context,
        );

        print('Server response: $response');

        if (response['success'] == true) {
          print('Registration successful for: ${_emailController.text}');
          ToastHelper.showSuccess(
            context,
            'Registration successful! Please verify your email.',
          );

          Navigator.push(
            context,
            MaterialPageRoute(
              builder: (context) =>
                  OTPVerificationPage(email: _emailController.text),
            ),
          );
        } else {
          setState(() {
            _errorMessage =
                response['data']?.toString().replaceAll(RegExp(r'[{}]'), '') ??
                    'Registration failed. Please try again.';
          });
        }
      } catch (e) {
        print('Error during registration: ${e.toString()}');
        setState(() {
          _errorMessage =
              'Unable to complete the registration. Please try again later.';
        });
      } finally {
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

    return Form(
      key: _formKey,
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          TextFormField(
            controller: _emailController,
            keyboardType: TextInputType.emailAddress,
            decoration: AppStyles.inputDecoration.copyWith(
              hintText: 'Email',
              prefixIcon: Icon(
                Icons.email_outlined,
                color: isDarkMode ? Colors.white : Colors.black,
              ),
              hintStyle: TextStyle(
                color: isDarkMode ? Colors.white70 : Colors.black54,
              ),
            ),
            validator: (value) {
              if (value == null || value.isEmpty) {
                return 'Please enter your email';
              }
              final emailRegex = RegExp(
                r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
              );
              if (!emailRegex.hasMatch(value)) {
                return 'Please enter a valid email address';
              }
              return null;
            },
          ),
          const SizedBox(height: 16),
          TextFormField(
            controller: _usernameController,
            decoration: AppStyles.inputDecoration.copyWith(
              hintText: 'Username',
              prefixIcon: Icon(
                Icons.person_outline,
                color: isDarkMode ? Colors.white : Colors.black,
              ),
              hintStyle: TextStyle(
                color: isDarkMode ? Colors.white70 : Colors.black54,
              ),
            ),
            validator: (value) {
              if (value == null || value.isEmpty) {
                return 'Please enter your username';
              }
              return null;
            },
          ),
          const SizedBox(height: 16),
          TextFormField(
            controller: _passwordController,
            obscureText: _obscurePassword,
            decoration: AppStyles.inputDecoration.copyWith(
              hintText: 'Password',
              prefixIcon: Icon(
                Icons.lock_outline,
                color: isDarkMode ? Colors.white : Colors.black,
              ),
              suffixIcon: IconButton(
                icon: Icon(
                  _obscurePassword ? Icons.visibility_off : Icons.visibility,
                  color: isDarkMode ? Colors.white : Colors.black,
                ),
                onPressed: () =>
                    setState(() => _obscurePassword = !_obscurePassword),
              ),
              hintStyle: TextStyle(
                color: isDarkMode ? Colors.white70 : Colors.black54,
              ),
            ),
            validator: (value) {
              if (value == null || value.isEmpty) {
                return 'Please enter your password';
              }
              if (value.length < 6) {
                return 'Password must be at least 6 characters';
              }
              return null;
            },
          ),
          const SizedBox(height: 16),
          TextFormField(
            controller: _confirmPasswordController,
            obscureText: _obscureConfirmPassword,
            decoration: AppStyles.inputDecoration.copyWith(
              hintText: 'Confirm Password',
              prefixIcon: Icon(
                Icons.lock_outline,
                color: isDarkMode ? Colors.white : Colors.black,
              ),
              suffixIcon: IconButton(
                icon: Icon(
                  _obscureConfirmPassword
                      ? Icons.visibility_off
                      : Icons.visibility,
                  color: isDarkMode ? Colors.white : Colors.black,
                ),
                onPressed: () => setState(
                  () => _obscureConfirmPassword = !_obscureConfirmPassword,
                ),
              ),
              hintStyle: TextStyle(
                color: isDarkMode ? Colors.white70 : Colors.black54,
              ),
            ),
            validator: (value) {
              if (value == null || value.isEmpty) {
                return 'Please confirm your password';
              }
              if (value != _passwordController.text) {
                return 'Passwords do not match';
              }
              return null;
            },
          ),
          const SizedBox(height: 24),

          // Display error message if there is one
          if (_errorMessage.isNotEmpty)
            ErrorMessageBox(errorMessage: _errorMessage),

          SizedBox(
            width: double.infinity,
            height: 50,
            child: _isLoading
                ? const Center(child: CircularProgressIndicator())
                : ElevatedButton(
                    onPressed: _handleSignup,
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
                      'Sign Up',
                      style: TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ),
          ),
        ],
      ),
    );
  }
}
