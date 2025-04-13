import 'dart:async';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:echo_journal1/services/auth/login_service.dart';
import 'package:echo_journal1/core/configs/theme/app-styles.dart';
import 'package:echo_journal1/features/widgets/navbar.dart';
import 'package:echo_journal1/services/auth/session_management.dart';
import 'package:echo_journal1/core/configs/theme/gradient-bg-pattern.dart';
import 'package:echo_journal1/core/configs/theme/theme-provider.dart';
import 'package:provider/provider.dart';

class Verify2FALoginPage extends StatefulWidget {
  final String email;

  const Verify2FALoginPage({Key? key, required this.email}) : super(key: key);

  @override
  _Verify2FALoginPageState createState() => _Verify2FALoginPageState();
}

class _Verify2FALoginPageState extends State<Verify2FALoginPage>
    with SingleTickerProviderStateMixin {
  final List<TextEditingController> _controllers =
      List.generate(6, (index) => TextEditingController());
  final List<FocusNode> _focusNodes = List.generate(6, (index) => FocusNode());
  bool _isLoading = false;
  String _errorMessage = '';
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

  void _startResendTimer() {
    _timer = Timer.periodic(const Duration(seconds: 1), (timer) {
      if (_resendTimer > 0) {
        setState(() {
          _resendTimer--;
        });
      } else {
        _timer?.cancel();
      }
    });
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

  Future<void> _verify2FA() async {
    final otp = _controllers.map((c) => c.text).join();
    if (otp.length != 6) return;

    setState(() {
      _isLoading = true;
      _errorMessage = '';
    });

    try {
      final response = await AuthService.verify2FALogin(
        widget.email,
        otp,
      );

      if (mounted) {
        if (response['success']) {
          // Start session before navigation
          await SessionManager.startSession();

          // Navigate to home page
          Navigator.of(context).pushAndRemoveUntil(
            MaterialPageRoute(builder: (context) => const NavBar()),
            (route) => false,
          );
        } else {
          setState(() {
            _errorMessage = response['error'];
          });
        }
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _errorMessage = e.toString();
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

  Future<void> _resendCode() async {
    if (_resendTimer > 0) return;

    setState(() {
      _isLoading = true;
      _errorMessage = '';
    });

    try {
      final response = await AuthService.resend2FACode(widget.email);

      if (mounted) {
        if (response['success']) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text(response['message']),
              backgroundColor: Colors.green,
            ),
          );
          setState(() {
            _resendTimer = 30;
          });
          _startResendTimer();
        } else {
          setState(() {
            _errorMessage = response['error'];
          });
        }
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _errorMessage = e.toString();
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
                                Icons.security_rounded,
                                size: 48,
                                color: isDarkMode
                                    ? Colors.white
                                    : const Color(0xFFFF758C),
                              ),
                            ),
                            const SizedBox(height: 16),
                            Text(
                              'Two-Factor Authentication',
                              style: TextStyle(
                                fontSize: 24,
                                fontWeight: FontWeight.bold,
                                color:
                                    isDarkMode ? Colors.white : Colors.black87,
                              ),
                            ),
                            const SizedBox(height: 8),
                            Text(
                              'Enter the verification code sent to\n${widget.email}',
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
                          children: [
                            Row(
                              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                              children: List.generate(
                                6,
                                (index) => SizedBox(
                                  width: 40,
                                  child: TextField(
                                    controller: _controllers[index],
                                    focusNode: _focusNodes[index],
                                    decoration: InputDecoration(
                                      counterText: '',
                                      fillColor: isDarkMode
                                          ? Colors.grey[800]
                                          : Colors.grey[50],
                                      filled: true,
                                      border: OutlineInputBorder(
                                        borderRadius: BorderRadius.circular(10),
                                        borderSide: BorderSide(
                                          color: isDarkMode
                                              ? Colors.grey[700]!
                                              : Colors.grey[300]!,
                                        ),
                                      ),
                                      enabledBorder: OutlineInputBorder(
                                        borderRadius: BorderRadius.circular(10),
                                        borderSide: BorderSide(
                                          color: isDarkMode
                                              ? Colors.grey[700]!
                                              : Colors.grey[300]!,
                                        ),
                                      ),
                                      focusedBorder: OutlineInputBorder(
                                        borderRadius: BorderRadius.circular(10),
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
                                        _verify2FA();
                                      }
                                    },
                                  ),
                                ),
                              ),
                            ),
                            if (_errorMessage.isNotEmpty) ...[
                              const SizedBox(height: 16),
                              Text(
                                _errorMessage,
                                style: const TextStyle(color: Colors.red),
                                textAlign: TextAlign.center,
                              ),
                            ],
                            const SizedBox(height: 24),
                            SizedBox(
                              width: double.infinity,
                              height: 50,
                              child: _isLoading
                                  ? const Center(
                                      child: CircularProgressIndicator())
                                  : ElevatedButton(
                                      onPressed: _verify2FA,
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
                            TextButton(
                              onPressed: _resendTimer > 0 ? null : _resendCode,
                              child: Text(
                                _resendTimer > 0
                                    ? 'Resend code in $_resendTimer seconds'
                                    : 'Resend Code',
                                style: TextStyle(
                                  color: _resendTimer > 0
                                      ? Colors.grey
                                      : const Color(0xFFFF758C),
                                ),
                              ),
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
}
