import 'package:echo_journal1/common/widgets/error_message.dart';
import 'package:echo_journal1/core/configs/theme/app-styles.dart';
import 'package:echo_journal1/core/configs/theme/theme-provider.dart';
import 'package:echo_journal1/features/authentication/pages/otp_verification_page.dart';
import 'package:echo_journal1/services/auth/register_service.dart';
import 'package:echo_journal1/utils/toast_helper.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:password_strength/password_strength.dart';

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
  double _passwordStrength = 0.0;
  String _passwordStrengthText = '';
  Color _passwordStrengthColor = Colors.grey;

  @override
  void dispose() {
    _emailController.dispose();
    _usernameController.dispose();
    _passwordController.dispose();
    _confirmPasswordController.dispose();
    super.dispose();
  }

  void _updatePasswordStrength(String password) {
    if (password.isEmpty) {
      setState(() {
        _passwordStrength = 0.0;
        _passwordStrengthText = '';
        _passwordStrengthColor = Colors.grey;
      });
      return;
    }

    final strength = estimatePasswordStrength(password);
    setState(() {
      _passwordStrength = strength;
      if (strength < 0.3) {
        _passwordStrengthText = 'Weak';
        _passwordStrengthColor = Colors.red;
      } else if (strength < 0.7) {
        _passwordStrengthText = 'Medium';
        _passwordStrengthColor = Colors.orange;
      } else {
        _passwordStrengthText = 'Strong';
        _passwordStrengthColor = Colors.green;
      }
    });
  }

  Future<void> _handleSignup() async {
    if (_formKey.currentState!.validate()) {
      setState(() {
        _isLoading = true;
        _errorMessage = '';
      });

      try {
        print("Starting registration process...");
        final Map<String, dynamic> response = await RegisterService.register(
          _usernameController.text,
          _passwordController.text,
          _emailController.text,
          context,
        );

        if (mounted) {
          print("Registration response received: $response");

          if (response['success'] == true) {
            print(
                "Registration successful, showing toast and navigating to OTP page");

            // Show success message
            ToastHelper.showSuccess(
              context,
              response['message'] ??
                  'Registration initiated! Please check your email for OTP.',
            );

            // Clear form
            _emailController.clear();
            _usernameController.clear();
            _passwordController.clear();
            _confirmPasswordController.clear();

            // Navigate to OTP verification page
            Navigator.pushReplacement(
              context,
              MaterialPageRoute(
                builder: (context) => OTPVerificationPage(
                  email: response['email'] ?? _emailController.text,
                ),
              ),
            );
          } else {
            print("Registration failed: ${response['error']}");
            setState(() {
              _errorMessage =
                  response['error'] ?? 'Registration failed. Please try again.';
            });
            ToastHelper.showError(context, _errorMessage);
          }
        }
      } catch (e) {
        print("Error during registration: $e");
        if (mounted) {
          setState(() {
            _errorMessage =
                'Unable to complete registration. Please try again later.';
          });
          ToastHelper.showError(context, _errorMessage);
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

    return Form(
      key: _formKey,
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          _buildTextField(
            controller: _emailController,
            label: 'Email',
            icon: Icons.email_outlined,
            keyboardType: TextInputType.emailAddress,
            validator: (value) {
              if (value == null || value.isEmpty) {
                return 'Please enter your email';
              }
              final emailRegex = RegExp(
                r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
              );
              if (!emailRegex.hasMatch(value)) {
                return 'Please enter a valid email address';
              }
              if (value.length > 255) {
                return 'Email address is too long';
              }
              return null;
            },
            isDarkMode: isDarkMode,
          ),
          const SizedBox(height: 16),
          _buildTextField(
            controller: _usernameController,
            label: 'Username',
            icon: Icons.person_outline,
            validator: (value) {
              if (value == null || value.isEmpty) {
                return 'Please enter your username';
              }
              if (value.length < 3) {
                return 'Username must be at least 3 characters';
              }
              if (value.length > 30) {
                return 'Username cannot exceed 30 characters';
              }
              if (!RegExp(r'^[a-zA-Z._]+$').hasMatch(value)) {
                return 'Username can only contain letters, dots, and underscores';
              }
              if (value.startsWith('.') ||
                  value.startsWith('_') ||
                  value.endsWith('.') ||
                  value.endsWith('_')) {
                return 'Username cannot start or end with dots or underscores';
              }
              if (value.contains('..') || value.contains('__')) {
                return 'Username cannot contain consecutive dots or underscores';
              }
              return null;
            },
            isDarkMode: isDarkMode,
          ),
          const SizedBox(height: 16),
          _buildTextField(
            controller: _passwordController,
            label: 'Password',
            icon: Icons.lock_outline,
            obscureText: _obscurePassword,
            onChanged: _updatePasswordStrength,
            validator: (value) {
              if (value == null || value.isEmpty) {
                return 'Please enter your password';
              }
              if (value.length < 8) {
                return 'Password must be at least 8 characters';
              }
              if (value.length > 128) {
                return 'Password cannot exceed 128 characters';
              }
              if (!RegExp(r'[A-Z]').hasMatch(value)) {
                return 'Password must contain at least one uppercase letter';
              }
              if (!RegExp(r'[a-z]').hasMatch(value)) {
                return 'Password must contain at least one lowercase letter';
              }
              if (!RegExp(r'[0-9]').hasMatch(value)) {
                return 'Password must contain at least one number';
              }
              if (!RegExp(r'[!@#$%^&*(),.?":{}|<>]').hasMatch(value)) {
                return 'Password must contain at least one special character';
              }
              if (value.contains(RegExp(r'\s'))) {
                return 'Password cannot contain spaces';
              }
              return null;
            },
            isDarkMode: isDarkMode,
            suffixIcon: IconButton(
              icon: Icon(
                _obscurePassword ? Icons.visibility_off : Icons.visibility,
                color: isDarkMode ? Colors.white : Colors.black,
              ),
              onPressed: () =>
                  setState(() => _obscurePassword = !_obscurePassword),
            ),
          ),
          if (_passwordController.text.isNotEmpty) ...[
            const SizedBox(height: 8),
            LinearProgressIndicator(
              value: _passwordStrength,
              backgroundColor: Colors.grey[300],
              valueColor: AlwaysStoppedAnimation<Color>(_passwordStrengthColor),
              minHeight: 4,
            ),
            const SizedBox(height: 4),
            Text(
              _passwordStrengthText,
              style: TextStyle(
                color: _passwordStrengthColor,
                fontSize: 12,
              ),
            ),
          ],
          const SizedBox(height: 16),
          _buildTextField(
            controller: _confirmPasswordController,
            label: 'Confirm Password',
            icon: Icons.lock_outline,
            obscureText: _obscureConfirmPassword,
            validator: (value) {
              if (value == null || value.isEmpty) {
                return 'Please confirm your password';
              }
              if (value != _passwordController.text) {
                return 'Passwords do not match';
              }
              return null;
            },
            isDarkMode: isDarkMode,
            suffixIcon: IconButton(
              icon: Icon(
                _obscureConfirmPassword
                    ? Icons.visibility_off
                    : Icons.visibility,
                color: isDarkMode ? Colors.white : Colors.black,
              ),
              onPressed: () => setState(
                  () => _obscureConfirmPassword = !_obscureConfirmPassword),
            ),
          ),
          const SizedBox(height: 24),
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

  Widget _buildTextField({
    required TextEditingController controller,
    required String label,
    required IconData icon,
    required String? Function(String?) validator,
    required bool isDarkMode,
    bool obscureText = false,
    TextInputType? keyboardType,
    Widget? suffixIcon,
    void Function(String)? onChanged,
  }) {
    return TextFormField(
      controller: controller,
      obscureText: obscureText,
      keyboardType: keyboardType,
      onChanged: onChanged,
      decoration: AppStyles.inputDecoration.copyWith(
        hintText: label,
        prefixIcon: Icon(
          icon,
          color: isDarkMode ? Colors.white : Colors.black,
        ),
        suffixIcon: suffixIcon,
        hintStyle: TextStyle(
          color: isDarkMode ? Colors.white70 : Colors.black54,
        ),
        filled: true,
        fillColor: isDarkMode ? Colors.grey[850] : Colors.grey[50],
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(15),
          borderSide: BorderSide.none,
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(15),
          borderSide: BorderSide(
            color: isDarkMode ? Colors.grey[700]! : Colors.grey[300]!,
          ),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(15),
          borderSide: const BorderSide(
            color: Color(0xFFFF758C),
            width: 2,
          ),
        ),
      ),
      validator: validator,
    );
  }
}
