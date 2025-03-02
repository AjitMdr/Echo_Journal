import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../../../../services/auth/login_service.dart';
import '../../../../services/auth/session_management.dart';
import '../../../core/constants/app_styles.dart';
import '../../../core/widgets/error_message.dart';
import '../../../../themes/theme_provider.dart';
import '../../home/widget/navbar.dart';

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
        final success = await AuthService.login(
          _usernameController.text,
          _passwordController.text,
        );

        if (success && mounted) {
          await SessionManager.startSession(); // Start new session
          if (!mounted) return;

          Navigator.pushReplacement(
            context,
            MaterialPageRoute(
              builder: (context) => const NavBar(),
            ),
          );
        } else if (mounted) {
          setState(() {
            _errorMessage = 'Invalid username or password';
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
            'Hello',
            style: TextStyle(
              fontSize: 32,
              fontWeight: FontWeight.bold,
              color: isDarkMode ? Colors.white : const Color(0xFFFF758C),
            ),
          ),
          const SizedBox(height: 8),
          const Text(
            'Sign into your account',
            style: TextStyle(
              fontSize: 16,
              color: Colors.grey,
            ),
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
        ],
      ),
    );
  }
}
