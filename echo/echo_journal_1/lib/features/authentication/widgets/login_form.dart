import 'package:echo_journal1/common/widgets/error_message.dart';
import 'package:echo_journal1/core/configs/theme/app-styles.dart';
import 'package:echo_journal1/core/configs/theme/theme-provider.dart';
import 'package:echo_journal1/features/widgets/navbar.dart';
import 'package:echo_journal1/services/auth/login_service.dart';
import 'package:echo_journal1/services/auth/session_management.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:echo_journal1/features/authentication/pages/otp_verification_page.dart';
import 'package:echo_journal1/features/authentication/pages/verify_account_page.dart';
import 'package:echo_journal1/features/authentication/pages/verify_2fa_login_page.dart';

class LoginForm extends StatefulWidget {
  const LoginForm({super.key});

  @override
  _LoginFormState createState() => _LoginFormState();
}

class _LoginFormState extends State<LoginForm> {
  final _formKey = GlobalKey<FormState>();
  final _usernameController = TextEditingController();
  final _passwordController = TextEditingController();
  bool _obscurePassword = true;
  bool _isLoading = false;
  String _errorMessage = '';

  @override
  void dispose() {
    _usernameController.dispose();
    _passwordController.dispose();
    super.dispose();
  }

  Future<void> _handleLogin() async {
    if (_formKey.currentState!.validate()) {
      setState(() {
        _isLoading = true;
        _errorMessage = '';
      });

      try {
        final response = await AuthService.login(
          _usernameController.text,
          _passwordController.text,
        );

        if (!mounted) return;

        if (response['requires_2fa'] == true) {
          // Show a success message before navigating
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text(response['message'] ??
                  '2FA code has been sent to your email'),
              backgroundColor: Colors.green,
            ),
          );

          // Navigate to 2FA verification page
          Navigator.pushReplacement(
            context,
            MaterialPageRoute(
              builder: (context) => Verify2FALoginPage(
                email: response['email'],
              ),
            ),
          );
        } else if (response['access'] != null && response['refresh'] != null) {
          // Non-2FA user successful login
          await AuthService.saveTokens(response['access'], response['refresh']);
          await SessionManager.startSession();

          if (!mounted) return;

          Navigator.pushReplacement(
            context,
            MaterialPageRoute(builder: (context) => const NavBar()),
          );
        } else if (response['needs_verification'] == true) {
          // Account needs verification
          Navigator.pushReplacement(
            context,
            MaterialPageRoute(
              builder: (context) => OTPVerificationPage(
                email: response['email'] ?? _usernameController.text,
              ),
            ),
          );
        } else {
          setState(() {
            _errorMessage =
                response['error']?.toString() ?? 'Authentication failed';
          });
        }
      } catch (e) {
        if (mounted) {
          setState(() {
            _errorMessage = 'Error: ${e.toString()}';
          });
        }
      } finally {
        if (mounted) {
          setState(() => _isLoading = false);
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
          Text(
            'Echo',
            style: TextStyle(
              fontSize: 32,
              fontWeight: FontWeight.bold,
              color: isDarkMode ? Colors.white : const Color(0xFFFF758C),
            ),
          ),
          const SizedBox(height: 8),
          const Text(
            'Sign into your account',
            style: TextStyle(fontSize: 16, color: Colors.grey),
          ),
          const SizedBox(height: 32),
          TextFormField(
            controller: _usernameController,
            keyboardType: TextInputType.emailAddress,
            autofillHints: const [AutofillHints.username],
            decoration: AppStyles.inputDecoration.copyWith(
              hintText: 'Username',
              hintStyle: TextStyle(
                color: isDarkMode
                    ? Colors.white.withOpacity(0.7)
                    : Colors.black.withOpacity(0.7),
              ),
              prefixIcon: Icon(
                Icons.person_outline,
                color: isDarkMode ? Colors.white : Colors.black,
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
            keyboardType: TextInputType.visiblePassword,
            autofillHints: const [AutofillHints.password],
            decoration: AppStyles.inputDecoration.copyWith(
              hintText: 'Password',
              hintStyle: TextStyle(
                color: isDarkMode
                    ? Colors.white.withOpacity(0.7)
                    : Colors.black.withOpacity(0.7),
              ),
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
            ),
            validator: (value) {
              if (value == null || value.isEmpty) {
                return 'Please enter your password';
              }
              return null;
            },
            textInputAction: TextInputAction.done,
            onFieldSubmitted: (_) => _handleLogin(),
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
                    onPressed: _handleLogin,
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
                      'Login',
                      style: TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ),
          ),
          const SizedBox(height: 16),
          TextButton(
            onPressed: () {
              Navigator.push(
                context,
                MaterialPageRoute(
                  builder: (context) => VerifyAccountPage(
                    email: _usernameController.text,
                  ),
                ),
              );
            },
            child: Text(
              'Need to verify your account?',
              style: TextStyle(
                color: isDarkMode ? Colors.white70 : const Color(0xFFFF758C),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
