import 'dart:async';
import 'package:echo_journal1/core/configs/theme/gradient-bg-pattern.dart';
import 'package:echo_journal1/core/configs/theme/theme-provider.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'login_page.dart';
import 'package:echo_journal1/utils/toast_helper.dart';
import 'package:echo_journal1/common/widgets/error_message.dart';
import 'package:echo_journal1/services/auth/forgot_password_service.dart';
import 'package:flutter/services.dart';

class ResetPasswordPage extends StatefulWidget {
  final String email;

  const ResetPasswordPage({super.key, required this.email});

  @override
  _ResetPasswordPageState createState() => _ResetPasswordPageState();
}

class _ResetPasswordPageState extends State<ResetPasswordPage> {
  final _formKey = GlobalKey<FormState>();
  final _otpController = TextEditingController();
  final _passwordController = TextEditingController();
  final _confirmPasswordController = TextEditingController();
  bool _isLoading = false;
  String _errorMessage = '';
  int _timerSeconds = 60;
  Timer? _timer;
  bool _canResendOTP = false;
  bool _showPassword = false;
  bool _showConfirmPassword = false;

  @override
  void initState() {
    super.initState();
    _startTimer();
  }

  @override
  void dispose() {
    _otpController.dispose();
    _passwordController.dispose();
    _confirmPasswordController.dispose();
    _timer?.cancel();
    super.dispose();
  }

  void _startTimer() {
    setState(() {
      _timerSeconds = 30;
      _canResendOTP = false;
    });
    _timer = Timer.periodic(const Duration(seconds: 1), (timer) {
      if (_timerSeconds > 0) {
        setState(() {
          _timerSeconds--;
        });
      } else {
        timer.cancel();
        setState(() {
          _canResendOTP = true;
        });
      }
    });
  }

  String? _validatePassword(String? value) {
    if (value == null || value.isEmpty) {
      return 'Please enter a password';
    }
    if (value.length < 8) {
      return 'Password must be at least 8 characters long';
    }
    if (!value.contains(RegExp(r'[A-Z]'))) {
      return 'Password must contain at least one uppercase letter';
    }
    if (!value.contains(RegExp(r'[a-z]'))) {
      return 'Password must contain at least one lowercase letter';
    }
    if (!value.contains(RegExp(r'[0-9]'))) {
      return 'Password must contain at least one number';
    }
    return null;
  }

  String? _validateConfirmPassword(String? value) {
    if (value == null || value.isEmpty) {
      return 'Please confirm your password';
    }
    if (value != _passwordController.text) {
      return 'Passwords do not match';
    }
    return null;
  }

  String? _validateOTP(String? value) {
    if (value == null || value.isEmpty) {
      return 'Please enter the OTP';
    }
    if (value.length != 6) {
      return 'OTP must be 6 digits';
    }
    if (!RegExp(r'^[0-9]+$').hasMatch(value)) {
      return 'OTP must contain only numbers';
    }
    return null;
  }

  Future<void> _resetPassword() async {
    if (_formKey.currentState!.validate()) {
      setState(() {
        _isLoading = true;
        _errorMessage = '';
      });

      try {
        final result = await ForgotPasswordService.verifyOTPAndResetPassword(
          email: widget.email,
          otp: _otpController.text,
          newPassword: _passwordController.text,
        );

        if (mounted) {
          if (result['success'] == true) {
            ToastHelper.showSuccess(
                context, result['message'] ?? 'Password reset successful');
            Navigator.of(context).pushAndRemoveUntil(
              MaterialPageRoute(builder: (context) => const LoginPage()),
              (route) => false,
            );
          } else {
            setState(() {
              _errorMessage = result['error'] ?? 'Failed to reset password';
              // If OTP is invalid or expired, enable resend button immediately
              if (_errorMessage.contains('OTP') &&
                      _errorMessage.contains('expired') ||
                  _errorMessage.contains('Invalid OTP')) {
                _timerSeconds = 0;
                _canResendOTP = true;
                if (_timer != null) {
                  _timer!.cancel();
                }
              }
            });
          }
        }
      } catch (e) {
        if (mounted) {
          setState(() {
            _errorMessage = 'An error occurred. Please try again.';
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

  Future<void> _resendOTP() async {
    if (!_canResendOTP) return;

    setState(() {
      _isLoading = true;
      _errorMessage = '';
    });

    try {
      final result =
          await ForgotPasswordService.resendPasswordResetOTP(widget.email);

      if (mounted) {
        if (result['success'] == true) {
          _otpController.clear(); // Clear the OTP field when resending
          ToastHelper.showSuccess(
              context, result['message'] ?? 'New OTP sent successfully');
          _startTimer();
        } else {
          setState(() {
            _errorMessage = result['error'] ?? 'Failed to resend OTP';
          });
        }
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _errorMessage = 'An error occurred. Please try again.';
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
    final size = MediaQuery.of(context).size;

    return Scaffold(
      backgroundColor: Color(0xFFFF7676),
      body: Container(
        height: size.height,
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [
              Color(0xFFFF8585), // Lighter pink at top
              Color(0xFFFF9999), // Mid pink
              Color(0xFFFFB5B5), // Softer pink at bottom
            ],
            stops: [0.0, 0.3, 1.0],
          ),
        ),
        child: SafeArea(
          child: SingleChildScrollView(
            physics: ClampingScrollPhysics(),
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 24.0),
              child: Form(
                key: _formKey,
                child: Column(
                  children: [
                    SizedBox(height: size.height * 0.08),
                    Text(
                      'Reset Password',
                      style: TextStyle(
                        fontSize: 24,
                        fontWeight: FontWeight.w600,
                        color: Colors.black87,
                        letterSpacing: 0.3,
                      ),
                    ),
                    const SizedBox(height: 12),
                    Text(
                      'Enter the OTP sent to',
                      style: TextStyle(
                        fontSize: 14,
                        color: Colors.black54,
                        letterSpacing: 0.2,
                      ),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      widget.email,
                      style: TextStyle(
                        fontSize: 14,
                        color: Colors.black87,
                        fontWeight: FontWeight.w500,
                        letterSpacing: 0.2,
                      ),
                    ),
                    const SizedBox(height: 40),

                    // OTP Input
                    Container(
                      decoration: BoxDecoration(
                        color: Colors.white,
                        borderRadius: BorderRadius.circular(30),
                        boxShadow: [
                          BoxShadow(
                            color: Colors.black.withOpacity(0.05),
                            blurRadius: 10,
                            offset: Offset(0, 2),
                          ),
                        ],
                      ),
                      child: TextFormField(
                        controller: _otpController,
                        decoration: InputDecoration(
                          hintText: 'OTP',
                          hintStyle: TextStyle(
                            color: Colors.black38,
                            fontSize: 16,
                            fontWeight: FontWeight.w400,
                          ),
                          border: OutlineInputBorder(
                            borderRadius: BorderRadius.circular(30),
                            borderSide: BorderSide.none,
                          ),
                          contentPadding: EdgeInsets.symmetric(
                              horizontal: 24, vertical: 16),
                          suffixText: '${_otpController.text.length}/6',
                          suffixStyle: TextStyle(
                            color: Colors.black38,
                            fontSize: 12,
                          ),
                        ),
                        keyboardType: TextInputType.number,
                        maxLength: 6,
                        inputFormatters: [
                          FilteringTextInputFormatter.digitsOnly
                        ],
                        validator: _validateOTP,
                        style: TextStyle(
                          color: Colors.black87,
                          fontSize: 15,
                          fontWeight: FontWeight.w400,
                        ),
                        buildCounter: (context,
                                {required currentLength,
                                required isFocused,
                                maxLength}) =>
                            null,
                        onChanged: (value) => setState(() {}),
                      ),
                    ),

                    const SizedBox(height: 16),

                    // New Password Input
                    Container(
                      decoration: BoxDecoration(
                        color: Colors.white,
                        borderRadius: BorderRadius.circular(30),
                        boxShadow: [
                          BoxShadow(
                            color: Colors.black.withOpacity(0.05),
                            blurRadius: 10,
                            offset: Offset(0, 2),
                          ),
                        ],
                      ),
                      child: TextFormField(
                        controller: _passwordController,
                        obscureText: !_showPassword,
                        decoration: InputDecoration(
                          hintText: 'New Password',
                          hintStyle: TextStyle(
                            color: Colors.black38,
                            fontSize: 16,
                            fontWeight: FontWeight.w400,
                          ),
                          border: OutlineInputBorder(
                            borderRadius: BorderRadius.circular(30),
                            borderSide: BorderSide.none,
                          ),
                          contentPadding: EdgeInsets.symmetric(
                              horizontal: 24, vertical: 16),
                          suffixIcon: Padding(
                            padding: const EdgeInsets.only(right: 12),
                            child: IconButton(
                              icon: Icon(
                                _showPassword
                                    ? Icons.visibility_off
                                    : Icons.visibility,
                                color: Colors.black38,
                                size: 22,
                              ),
                              onPressed: () => setState(
                                  () => _showPassword = !_showPassword),
                            ),
                          ),
                        ),
                        validator: _validatePassword,
                        style: TextStyle(
                          color: Colors.black87,
                          fontSize: 15,
                          fontWeight: FontWeight.w400,
                        ),
                      ),
                    ),

                    const SizedBox(height: 16),

                    // Confirm Password Input
                    Container(
                      decoration: BoxDecoration(
                        color: Colors.white,
                        borderRadius: BorderRadius.circular(30),
                        boxShadow: [
                          BoxShadow(
                            color: Colors.black.withOpacity(0.05),
                            blurRadius: 10,
                            offset: Offset(0, 2),
                          ),
                        ],
                      ),
                      child: TextFormField(
                        controller: _confirmPasswordController,
                        obscureText: !_showConfirmPassword,
                        decoration: InputDecoration(
                          hintText: 'Confirm Password',
                          hintStyle: TextStyle(
                            color: Colors.black38,
                            fontSize: 16,
                            fontWeight: FontWeight.w400,
                          ),
                          border: OutlineInputBorder(
                            borderRadius: BorderRadius.circular(30),
                            borderSide: BorderSide.none,
                          ),
                          contentPadding: EdgeInsets.symmetric(
                              horizontal: 24, vertical: 16),
                          suffixIcon: Padding(
                            padding: const EdgeInsets.only(right: 12),
                            child: IconButton(
                              icon: Icon(
                                _showConfirmPassword
                                    ? Icons.visibility_off
                                    : Icons.visibility,
                                color: Colors.black38,
                                size: 22,
                              ),
                              onPressed: () => setState(() =>
                                  _showConfirmPassword = !_showConfirmPassword),
                            ),
                          ),
                        ),
                        validator: _validateConfirmPassword,
                        style: TextStyle(
                          color: Colors.black87,
                          fontSize: 15,
                          fontWeight: FontWeight.w400,
                        ),
                      ),
                    ),

                    const SizedBox(height: 24),

                    // Reset Password Button
                    Container(
                      width: double.infinity,
                      height: 52,
                      decoration: BoxDecoration(
                        color: Color(0xFFFF5C5C),
                        borderRadius: BorderRadius.circular(30),
                        boxShadow: [
                          BoxShadow(
                            color: Color(0xFFFF5C5C).withOpacity(0.3),
                            blurRadius: 8,
                            offset: Offset(0, 2),
                          ),
                        ],
                      ),
                      child: TextButton(
                        onPressed: _isLoading ? null : _resetPassword,
                        style: TextButton.styleFrom(
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(30),
                          ),
                          padding: EdgeInsets.zero,
                        ),
                        child: _isLoading
                            ? SizedBox(
                                height: 24,
                                width: 24,
                                child: CircularProgressIndicator(
                                  strokeWidth: 2.5,
                                  valueColor: AlwaysStoppedAnimation<Color>(
                                      Colors.white),
                                ),
                              )
                            : Text(
                                'Reset Password',
                                style: TextStyle(
                                  color: Colors.white,
                                  fontSize: 16,
                                  fontWeight: FontWeight.w500,
                                  letterSpacing: 0.5,
                                ),
                              ),
                      ),
                    ),

                    const SizedBox(height: 16),

                    // Resend OTP Button
                    Center(
                      child: TextButton(
                        onPressed:
                            (_canResendOTP && !_isLoading) ? _resendOTP : null,
                        style: TextButton.styleFrom(
                          foregroundColor: Colors.white,
                          padding:
                              EdgeInsets.symmetric(vertical: 8, horizontal: 16),
                          disabledForegroundColor:
                              Colors.white.withOpacity(0.7),
                        ),
                        child: Text(
                          _canResendOTP
                              ? 'Resend OTP'
                              : 'Resend OTP in $_timerSeconds seconds',
                          style: TextStyle(
                            fontSize: 14,
                            fontWeight: FontWeight.w400,
                            color: Colors.white.withOpacity(0.9),
                            letterSpacing: 0.3,
                          ),
                        ),
                      ),
                    ),
                    SizedBox(height: 16), // Add bottom padding
                  ],
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }
}
