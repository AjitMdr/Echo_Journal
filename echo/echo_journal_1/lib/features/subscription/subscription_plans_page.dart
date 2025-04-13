import 'package:flutter/material.dart';
import '../../models/subscription/plan.dart';
import '../../services/subscription/subscription_service.dart';
import '../../core/configs/theme/theme-provider.dart';
import '../../services/esewa/esewa.dart';
import '../../core/providers/subscription_provider.dart';
import 'package:provider/provider.dart';

class SubscriptionPlansPage extends StatefulWidget {
  const SubscriptionPlansPage({Key? key}) : super(key: key);

  @override
  _SubscriptionPlansPageState createState() => _SubscriptionPlansPageState();
}

class _SubscriptionPlansPageState extends State<SubscriptionPlansPage> {
  final SubscriptionService _subscriptionService = SubscriptionService();
  List<Plan> _plans = [];
  bool _isLoading = true;
  String? _error;

  @override
  void initState() {
    super.initState();
    _checkSubscriptionAndLoad();
  }

  Future<void> _checkSubscriptionAndLoad() async {
    try {
      setState(() {
        _isLoading = true;
      });

      // Get subscription provider
      final subscriptionProvider =
          Provider.of<SubscriptionProvider>(context, listen: false);

      // Check if user has an active premium subscription
      final isPremium = subscriptionProvider.subscription?.status == 'ACTIVE' &&
          subscriptionProvider.subscription?.planDetails?.planType == 'PREMIUM';

      // If user has active premium subscription, go to home
      if (isPremium && mounted) {
        Navigator.pushNamedAndRemoveUntil(
          context,
          '/',
          (route) => false,
        );
        return;
      }

      // Otherwise load plans
      await _loadPlans();
    } catch (e) {
      if (mounted) {
        setState(() {
          _error = e.toString();
          _isLoading = false;
        });
      }
    }
  }

  Future<void> _loadPlans() async {
    try {
      final plans = await _subscriptionService.getPlans();
      if (mounted) {
        setState(() {
          _plans = plans;
          _isLoading = false;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _error = e.toString();
          _isLoading = false;
        });
      }
    }
  }

  Future<void> _handleSubscribe(Plan plan) async {
    try {
      setState(() => _isLoading = true);

      // Initialize eSewa payment
      final esewaService = Esewa();

      // Convert price to paisa (cents) for eSewa
      final priceInPaisa = (plan.price * 100).round();

      // Start eSewa payment process
      await esewaService.pay(
        productId: plan.id.toString(),
        productName: 'Premium Plan Subscription',
        productPrice: priceInPaisa.toString(),
        onSuccess: (result) async {
          try {
            setState(() {
              _isLoading = true;
            });

            final payment = await _subscriptionService.createPayment(
              plan.id.toString(),
              refId: result.refId,
            );

            if (payment.status == 'SUCCESS') {
              if (mounted) {
                ScaffoldMessenger.of(context).showSnackBar(
                  const SnackBar(
                    content: Text('Payment successful! Redirecting to home...'),
                    backgroundColor: Colors.green,
                    duration: Duration(seconds: 3),
                  ),
                );

                // Update subscription provider
                final subscriptionProvider =
                    Provider.of<SubscriptionProvider>(context, listen: false);
                await subscriptionProvider.refreshSubscription();

                // Navigate to home page and remove all previous routes
                Navigator.pushNamedAndRemoveUntil(
                  context,
                  '/',
                  (route) => false,
                );
              }
            }
          } catch (e) {
            if (mounted) {
              ScaffoldMessenger.of(context).showSnackBar(
                SnackBar(
                  content: Text('Payment verification failed: ${e.toString()}'),
                  backgroundColor: Colors.red,
                  duration: Duration(seconds: 5),
                ),
              );
              setState(() => _isLoading = false);
            }
          }
        },
        onFailure: (error) {
          if (mounted) {
            ScaffoldMessenger.of(context).showSnackBar(
              SnackBar(
                content: Text('Payment failed: $error'),
                backgroundColor: Colors.red,
                duration: Duration(seconds: 5),
              ),
            );
            setState(() => _isLoading = false);
          }
        },
        onCancelled: () {
          if (mounted) {
            ScaffoldMessenger.of(context).showSnackBar(
              const SnackBar(
                content: Text('Payment cancelled by user'),
                backgroundColor: Colors.grey,
                duration: Duration(seconds: 3),
              ),
            );
            setState(() => _isLoading = false);
          }
        },
      );
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Failed to initiate payment: ${e.toString()}'),
            backgroundColor: Colors.red,
            duration: Duration(seconds: 5),
          ),
        );
        setState(() => _isLoading = false);
      }
    }
  }

  Widget _buildPlanCard(Plan plan) {
    final themeProvider = Provider.of<ThemeProvider>(context);
    final isDarkMode = themeProvider.isDarkMode;
    final isPremium = plan.planType == 'PREMIUM';

    return Card(
      elevation: 8,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(15),
      ),
      child: Container(
        padding: const EdgeInsets.all(20),
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(15),
          gradient: isPremium
              ? const LinearGradient(
                  colors: [Color(0xFF6B4EFF), Color(0xFF9747FF)],
                  begin: Alignment.topLeft,
                  end: Alignment.bottomRight,
                )
              : null,
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              plan.name,
              style: TextStyle(
                fontSize: 24,
                fontWeight: FontWeight.bold,
                color: isPremium ? Colors.white : null,
              ),
            ),
            const SizedBox(height: 10),
            if (isPremium) ...[
              Row(
                crossAxisAlignment: CrossAxisAlignment.end,
                children: [
                  Text(
                    '\Rs ${plan.price.toStringAsFixed(2)}',
                    style: const TextStyle(
                      fontSize: 36,
                      fontWeight: FontWeight.bold,
                      color: Colors.white,
                    ),
                  ),
                  const SizedBox(width: 8),
                  const Padding(
                    padding: EdgeInsets.only(bottom: 6),
                    child: Text(
                      '/month',
                      style: TextStyle(
                        fontSize: 16,
                        color: Colors.white70,
                      ),
                    ),
                  ),
                ],
              ),
            ] else ...[
              Row(
                crossAxisAlignment: CrossAxisAlignment.end,
                children: [
                  Text(
                    'FREE',
                    style: TextStyle(
                      fontSize: 36,
                      fontWeight: FontWeight.bold,
                      color: isDarkMode ? Colors.white : Colors.black,
                    ),
                  ),
                  const SizedBox(width: 8),
                  Padding(
                    padding: const EdgeInsets.only(bottom: 6),
                    child: Text(
                      'Default Plan',
                      style: TextStyle(
                        fontSize: 16,
                        color: isDarkMode ? Colors.white70 : Colors.black54,
                      ),
                    ),
                  ),
                ],
              ),
            ],
            const SizedBox(height: 20),
            Text(
              'Features:',
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
                color: isPremium ? Colors.white : null,
              ),
            ),
            const SizedBox(height: 10),
            _buildFeatureItem('Journal Entries', true, isPremium),
            _buildFeatureItem('Connect with Friends', true, isPremium),
            _buildFeatureItem('Chat with Friends', true, isPremium),
            _buildFeatureItem(
              'AI-powered Mood Analysis',
              isPremium,
              isPremium,
              subtitle: !isPremium
                  ? 'Upgrade to premium for \Rs ${plan.price.toStringAsFixed(2)}/month'
                  : null,
            ),
            _buildFeatureItem('Analytics', true, isPremium),
            _buildFeatureItem('Streaks', true, isPremium),
            const SizedBox(height: 20),
            SizedBox(
              width: double.infinity,
              child: ElevatedButton(
                onPressed: isPremium ? () => _handleSubscribe(plan) : null,
                style: ElevatedButton.styleFrom(
                  padding: const EdgeInsets.symmetric(vertical: 16),
                  backgroundColor: isPremium ? Colors.white : Colors.grey[300],
                  foregroundColor:
                      isPremium ? const Color(0xFF6B4EFF) : Colors.grey[600],
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(10),
                  ),
                ),
                child: Text(
                  isPremium ? 'Subscribe Now' : 'Current Plan',
                  style: const TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildFeatureItem(String feature, bool isIncluded, bool isPremium,
      {String? subtitle}) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 5),
      child: Row(
        children: [
          Icon(
            isIncluded ? Icons.check_circle : Icons.cancel,
            color: isIncluded
                ? (isPremium
                    ? const Color.fromARGB(255, 0, 255, 0)
                    : Colors.green)
                : Colors.red,
            size: 20,
          ),
          const SizedBox(width: 10),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  feature,
                  style: TextStyle(
                    color: isPremium ? Colors.white : null,
                  ),
                ),
                if (subtitle != null)
                  Text(
                    subtitle,
                    style: TextStyle(
                      fontSize: 12,
                      color: isPremium ? Colors.white70 : Colors.grey,
                    ),
                  ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  String _getDurationText(int days) {
    switch (days) {
      case 30:
        return 'Monthly';
      case 90:
        return '3 Months';
      case 180:
        return '6 Months';
      default:
        return '$days days';
    }
  }

  String _formatFeatureName(String name) {
    // Convert snake_case to Title Case
    return name
        .split('_')
        .map((word) => word[0].toUpperCase() + word.substring(1).toLowerCase())
        .join(' ');
  }

  @override
  Widget build(BuildContext context) {
    final themeProvider = Provider.of<ThemeProvider>(context);
    final isDarkMode = themeProvider.isDarkMode;

    if (_isLoading) {
      return Scaffold(
        backgroundColor: isDarkMode ? Colors.grey[900] : Colors.grey[50],
        body: const Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              CircularProgressIndicator(),
              SizedBox(height: 16),
              Text(
                'Checking subscription status...',
                style: TextStyle(fontSize: 16),
              ),
            ],
          ),
        ),
      );
    }

    // Helper function to get a unique key for a plan
    String getPlanKey(Plan plan) => '${plan.planType}_${plan.durationDays}';

    return Scaffold(
      backgroundColor: isDarkMode ? Colors.grey[900] : Colors.grey[50],
      appBar: AppBar(
        title: const Text('Premium Features'),
        backgroundColor: isDarkMode ? Colors.grey[850] : Colors.white,
        elevation: 0,
        leading: IconButton(
          icon: Icon(Icons.arrow_back),
          onPressed: () => Navigator.pushNamedAndRemoveUntil(
            context,
            '/',
            (route) => false,
          ),
        ),
      ),
      body: _error != null
          ? Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Text(
                    'Error: $_error',
                    style: TextStyle(
                      color: isDarkMode ? Colors.white : Colors.black,
                    ),
                  ),
                  const SizedBox(height: 16),
                  ElevatedButton(
                    onPressed: _checkSubscriptionAndLoad,
                    child: const Text('Retry'),
                  ),
                ],
              ),
            )
          : SingleChildScrollView(
              child: Column(
                children: [
                  Container(
                    padding: const EdgeInsets.all(20),
                    color: isDarkMode ? Colors.grey[850] : Colors.white,
                    child: Column(
                      children: [
                        Text(
                          'Unlock Mood Analysis',
                          style: TextStyle(
                            fontSize: 28,
                            fontWeight: FontWeight.bold,
                            color: isDarkMode ? Colors.white : Colors.black,
                          ),
                        ),
                        const SizedBox(height: 10),
                        Text(
                          'Get insights into your emotional well-being',
                          style: TextStyle(
                            fontSize: 16,
                            color: isDarkMode ? Colors.white70 : Colors.black54,
                          ),
                        ),
                      ],
                    ),
                  ),
                  ..._plans
                      .where((plan) => plan.isActive)
                      // Group plans by type and duration, take the latest one
                      .fold<Map<String, Plan>>({}, (map, plan) {
                        final key = getPlanKey(plan);
                        if (!map.containsKey(key) ||
                            plan.createdAt.isAfter(map[key]!.createdAt)) {
                          map[key] = plan;
                        }
                        return map;
                      })
                      .values
                      .toList()
                      .also((plans) {
                        plans.sort((a, b) {
                          if (a.planType == 'FREE') return -1;
                          if (b.planType == 'FREE') return 1;
                          return 0;
                        });
                      })
                      .map((plan) => _buildPlanCard(plan))
                      .toList(),
                ],
              ),
            ),
    );
  }
}

extension ListExtension<T> on List<T> {
  List<T> also(void Function(List<T>) action) {
    action(this);
    return this;
  }
}
