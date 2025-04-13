import 'dart:async';
import 'package:echo_journal1/core/configs/theme/app-styles.dart';
import 'package:echo_journal1/core/configs/theme/gradient-bg-pattern.dart';
import 'package:echo_journal1/core/configs/theme/theme-provider.dart';
import 'package:echo_journal1/services/auth/otp_service.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:provider/provider.dart';
import 'login_page.dart';
import 'signup_page.dart';
import 'package:echo_journal1/utils/toast_helper.dart';

class OTPVerificationPage extends StatefulWidget {
  final String email;

  const OTPVerificationPage({super.key, required this.email});

  @override
  _OTPVerificationPageState createState() => _OTPVerificationPageState();
}

class _OTPVerificationPageState extends State<OTPVerificationPage>
    with SingleTickerProviderStateMixin {
  final List<TextEditingController> _controllers =
      List.generate(6, (index) => TextEditingController());
  final List<FocusNode> _focusNodes = List.generate(6, (index) => FocusNode());
  bool _isLoading = false;
  int _resendTimer = 30;
  Timer? _timer;
  late AnimationController _controller;
  late Animation<double> _fadeAnimation;
  late Animation<Offset> _slideAnimation;

  @override
  void initState() {
    super.initState();
    _startResendTimer();
    _setupAnimations();
  }

  void _setupAnimations() {
    _controller = AnimationController(
      duration: const Duration(milliseconds: 1000),
      vsync: this,
    );

    _fadeAnimation = Tween<double>(
      begin: 0.0,
      end: 1.0,
    ).animate(CurvedAnimation(
      parent: _controller,
      curve: Curves.easeIn,
    ));

    _slideAnimation = Tween<Offset>(
      begin: const Offset(0, 0.5),
      end: Offset.zero,
    ).animate(CurvedAnimation(
      parent: _controller,
      curve: Curves.easeOut,
    ));

    _controller.forward();
  }

  @override
  void dispose() {
    _timer?.cancel();
    _controller.dispose();
    for (var controller in _controllers) {
      controller.dispose();
    }
    for (var node in _focusNodes) {
      node.dispose();
    }
    super.dispose();
  }

  void _startResendTimer() {
    _timer?.cancel();
    setState(() => _resendTimer = 30);
    _timer = Timer.periodic(const Duration(seconds: 1), (timer) {
      setState(() {
        if (_resendTimer > 0) {
          _resendTimer--;
        } else {
          timer.cancel();
        }
      });
    });
  }

  String _getOTP() {
    return _controllers.map((controller) => controller.text).join();
  }

  void _handleKeyPress(int index, RawKeyEvent event) {
    if (event is RawKeyDownEvent) {
      if (event.logicalKey == LogicalKeyboardKey.backspace &&
          _controllers[index].text.isEmpty &&
          index > 0) {
        _focusNodes[index - 1].requestFocus();
        _controllers[index - 1].clear();
      }
    }
  }

  Future<void> _verifyOTP() async {
    setState(() => _isLoading = true);
    try {
      final otp = _getOTP();
      if (otp.length != 6) {
        ToastHelper.showError(context, 'Please enter a valid 6-digit OTP');
        return;
      }

      final success = await OTPService.verifyOTP(widget.email, otp);

      if (success && mounted) {
        ToastHelper.showSuccess(
          context,
          'Email verified successfully! Please login.',
        );
        Navigator.pushAndRemoveUntil(
          context,
          MaterialPageRoute(builder: (context) => const LoginPage()),
          (route) => false,
        );
      } else if (mounted) {
        ToastHelper.showError(context, 'Invalid OTP. Please try again.');
        for (var controller in _controllers) {
          controller.clear();
        }
        FocusScope.of(context).requestFocus(_focusNodes[0]);
      }
    } catch (e) {
      if (mounted) {
        ToastHelper.showError(context, 'Error: ${e.toString()}');
      }
    } finally {
      setState(() => _isLoading = false);
    }
  }

  Future<void> _resendOTP() async {
    if (_resendTimer > 0) return;

    setState(() => _isLoading = true);

    try {
      final success = await OTPService.resendOTP(widget.email);

      if (success['success'] != null && mounted) {
        ToastHelper.showSuccess(context, 'OTP resent successfully!');
        _startResendTimer();
      } else if (mounted) {
        ToastHelper.showError(
          context,
          'Failed to resend OTP. Please try again.',
        );
      }
    } catch (e) {
      if (mounted) {
        ToastHelper.showError(context, 'Error: ${e.toString()}');
      }
    } finally {
      setState(() => _isLoading = false);
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
              child: FadeTransition(
                opacity: _fadeAnimation,
                child: SlideTransition(
                  position: _slideAnimation,
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      // Logo and Title Section
                      Container(
                        margin: const EdgeInsets.only(bottom: 24),
                        child: Column(
                          children: [
                            Container(
                              padding: const EdgeInsets.all(16),
                              decoration: BoxDecoration(
                                shape: BoxShape.circle,
                                color: isDarkMode
                                    ? Colors.grey[850]
                                    : Colors.white,
                                boxShadow: [
                                  BoxShadow(
                                    color: isDarkMode
                                        ? Colors.black.withOpacity(0.3)
                                        : Colors.black.withOpacity(0.1),
                                    blurRadius: 10,
                                    offset: const Offset(0, 5),
                                  ),
                                ],
                              ),
                              child: Icon(
                                Icons.mark_email_unread_rounded,
                                size: 48,
                                color: isDarkMode
                                    ? Colors.white
                                    : const Color(0xFFFF758C),
                              ),
                            ),
                            const SizedBox(height: 16),
                            Text(
                              'Verify Your Email',
                              style: TextStyle(
                                fontSize: 24,
                                fontWeight: FontWeight.bold,
                                color:
                                    isDarkMode ? Colors.white : Colors.black87,
                              ),
                            ),
                            const SizedBox(height: 8),
                            Text(
                              'We\'ve sent a verification code to\n${widget.email}',
                              style: TextStyle(
                                fontSize: 16,
                                color: isDarkMode
                                    ? Colors.white70
                                    : Colors.black54,
                              ),
                              textAlign: TextAlign.center,
                            ),
                          ],
                        ),
                      ),

                      // OTP Input Section
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
                            Row(
                              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                              children: List.generate(
                                6,
                                (index) => SizedBox(
                                  width: 45,
                                  height: 55,
                                  child: RawKeyboardListener(
                                    focusNode: FocusNode(),
                                    onKey: (event) =>
                                        _handleKeyPress(index, event),
                                    child: TextFormField(
                                      controller: _controllers[index],
                                      focusNode: _focusNodes[index],
                                      decoration:
                                          AppStyles.inputDecoration.copyWith(
                                        contentPadding:
                                            const EdgeInsets.symmetric(
                                          vertical: 12,
                                          horizontal: 12,
                                        ),
                                        counterText: '',
                                        fillColor: isDarkMode
                                            ? Colors.grey[800]
                                            : Colors.grey[50],
                                        filled: true,
                                        border: OutlineInputBorder(
                                          borderRadius:
                                              BorderRadius.circular(10),
                                          borderSide: BorderSide(
                                            color: isDarkMode
                                                ? Colors.grey[700]!
                                                : Colors.grey[300]!,
                                          ),
                                        ),
                                        enabledBorder: OutlineInputBorder(
                                          borderRadius:
                                              BorderRadius.circular(10),
                                          borderSide: BorderSide(
                                            color: isDarkMode
                                                ? Colors.grey[700]!
                                                : Colors.grey[300]!,
                                          ),
                                        ),
                                        focusedBorder: OutlineInputBorder(
                                          borderRadius:
                                              BorderRadius.circular(10),
                                          borderSide: const BorderSide(
                                            color: Color(0xFFFF758C),
                                            width: 2,
                                          ),
                                        ),
                                      ),
                                      style: TextStyle(
                                        color: isDarkMode
                                            ? Colors.white
                                            : Colors.black,
                                        fontSize: 20,
                                        fontWeight: FontWeight.bold,
                                      ),
                                      textAlign: TextAlign.center,
                                      keyboardType: TextInputType.number,
                                      inputFormatters: [
                                        LengthLimitingTextInputFormatter(1),
                                        FilteringTextInputFormatter.digitsOnly,
                                      ],
                                      onChanged: (value) {
                                        if (value.isNotEmpty && index < 5) {
                                          _focusNodes[index + 1].requestFocus();
                                        }
                                        if (value.isNotEmpty && index == 5) {
                                          _verifyOTP();
                                        }
                                      },
                                    ),
                                  ),
                                ),
                              ),
                            ),
                            const SizedBox(height: 24),
                            SizedBox(
                              width: double.infinity,
                              height: 50,
                              child: _isLoading
                                  ? const Center(
                                      child: CircularProgressIndicator())
                                  : ElevatedButton(
                                      onPressed: _verifyOTP,
                                      style: ElevatedButton.styleFrom(
                                        backgroundColor: isDarkMode
                                            ? Colors.grey[800]
                                            : const Color(0xFFFF758C),
                                        elevation: 3,
                                        shape: RoundedRectangleBorder(
                                          borderRadius:
                                              BorderRadius.circular(15),
                                        ),
                                      ),
                                      child: const Text(
                                        'Verify',
                                        style: TextStyle(
                                          fontSize: 16,
                                          fontWeight: FontWeight.bold,
                                        ),
                                      ),
                                    ),
                            ),
                            const SizedBox(height: 16),
                            Row(
                              mainAxisAlignment: MainAxisAlignment.center,
                              children: [
                                Text(
                                  "Didn't receive the code?",
                                  style: TextStyle(
                                    color: isDarkMode
                                        ? Colors.white70
                                        : Colors.grey,
                                  ),
                                ),
                                TextButton(
                                  onPressed:
                                      _resendTimer > 0 ? null : _resendOTP,
                                  child: Text(
                                    _resendTimer > 0
                                        ? " Resend in $_resendTimer s"
                                        : " Resend",
                                    style: TextStyle(
                                      color: isDarkMode
                                          ? const Color(0xFFFF758C)
                                          : const Color(0xFFFF758C),
                                      fontWeight: FontWeight.bold,
                                    ),
                                  ),
                                ),
                              ],
                            ),
                          ],
                        ),
                      ),

                      // Help Section
                      Container(
                        margin: const EdgeInsets.only(top: 32),
                        padding: const EdgeInsets.all(16),
                        decoration: BoxDecoration(
                          color: isDarkMode ? Colors.grey[850] : Colors.white,
                          borderRadius: BorderRadius.circular(20),
                          boxShadow: [
                            BoxShadow(
                              color: isDarkMode
                                  ? Colors.black.withOpacity(0.3)
                                  : Colors.black.withOpacity(0.1),
                              blurRadius: 10,
                              offset: const Offset(0, 5),
                            ),
                          ],
                        ),
                        child: Column(
                          children: [
                            Text(
                              'Need Help?',
                              style: TextStyle(
                                fontSize: 18,
                                fontWeight: FontWeight.bold,
                                color:
                                    isDarkMode ? Colors.white : Colors.black87,
                              ),
                            ),
                            const SizedBox(height: 16),
                            _buildHelpItem(
                              'Check your spam folder',
                              Icons.folder_outlined,
                              isDarkMode,
                            ),
                            _buildHelpItem(
                              'Make sure the email is correct',
                              Icons.email_outlined,
                              isDarkMode,
                            ),
                            _buildHelpItem(
                              'Contact support if needed',
                              Icons.support_agent,
                              isDarkMode,
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
        ),
      ),
    );
  }

  Widget _buildHelpItem(String text, IconData icon, bool isDarkMode) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8),
      child: Row(
        children: [
          Icon(
            icon,
            color: const Color(0xFFFF758C),
            size: 20,
          ),
          const SizedBox(width: 12),
          Text(
            text,
            style: TextStyle(
              color: isDarkMode ? Colors.white70 : Colors.black87,
              fontSize: 14,
            ),
          ),
        ],
      ),
    );
  }
}
