import 'package:echo/page/features/home/widget/navbar.dart';
import 'package:flutter/material.dart';
import '../../../../services/login_service.dart';
import '../../home/home_page.dart';
import '../../../core/constants/app_styles.dart';
import '../../../core/widgets/error_message.dart'; // Import your ErrorMessageBox widget

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
  String _errorMessage = ''; // Error message state

  @override
  void dispose() {
    _usernameController.dispose();
    _passwordController.dispose();
    super.dispose();
  }

  Future<void> _handleLogin() async {
    if (_formKey.currentState!.validate()) {
      setState(() => _isLoading = true);
      setState(() => _errorMessage = ''); // Clear previous error message

      try {
        final success = await AuthService.login(
          _usernameController.text,
          _passwordController.text,
        );

        if (success && mounted) {
          Navigator.pushReplacement(
            context,
            MaterialPageRoute(
              builder: (context) =>
                  HomePage(username: _usernameController.text),
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
    return Form(
      key: _formKey,
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          const Text(
            'Hello',
            style: TextStyle(
              fontSize: 32,
              fontWeight: FontWeight.bold,
              color: Color(0xFFFF758C),
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
            keyboardType:
                TextInputType.emailAddress, // Assuming email-based login
            autofillHints: [AutofillHints.username],
            decoration: AppStyles.inputDecoration.copyWith(
              hintText: 'Username',
              prefixIcon: const Icon(Icons.person_outline),
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
            autofillHints: [AutofillHints.password],
            decoration: AppStyles.inputDecoration.copyWith(
              hintText: 'Password',
              prefixIcon: const Icon(Icons.lock_outline),
              suffixIcon: IconButton(
                icon: Icon(
                  _obscurePassword ? Icons.visibility_off : Icons.visibility,
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
            textInputAction:
                TextInputAction.done, // Allows submitting on "Done"
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
                    onPressed: _handleLogin,
                    style: ElevatedButton.styleFrom(
                      backgroundColor: const Color(0xFFFF758C),
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
