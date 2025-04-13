import 'package:flutter/material.dart';
import '../../models/subscription/subscription.dart';
import '../../core/providers/subscription_provider.dart';
import 'subscription_plans_page.dart';
import '../../core/configs/theme/theme-provider.dart';
import 'package:provider/provider.dart';

class SubscriptionCheckWidget extends StatefulWidget {
  final Widget child;
  final bool requirePremium;

  const SubscriptionCheckWidget({
    Key? key,
    required this.child,
    this.requirePremium = true,
  }) : super(key: key);

  @override
  _SubscriptionCheckWidgetState createState() =>
      _SubscriptionCheckWidgetState();
}

class _SubscriptionCheckWidgetState extends State<SubscriptionCheckWidget> {
  @override
  void initState() {
    super.initState();
    // Initial check will be handled by the build method
  }

  bool _hasAccess(SubscriptionProvider provider) {
    final subscription = provider.subscription;
    if (subscription == null) return false;

    // For premium features
    if (widget.requirePremium) {
      // Check both status and plan type
      return subscription.status == 'ACTIVE' &&
          subscription.planDetails?.planType == 'PREMIUM';
    }

    // For non-premium features, just check if subscription is active
    return subscription.status == 'ACTIVE';
  }

  @override
  Widget build(BuildContext context) {
    final isDarkMode = Provider.of<ThemeProvider>(context).isDarkMode;
    final subscriptionProvider = Provider.of<SubscriptionProvider>(context);

    if (subscriptionProvider.isLoading) {
      return Scaffold(
        backgroundColor: isDarkMode ? Colors.grey[900] : Colors.grey[50],
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const CircularProgressIndicator(),
              const SizedBox(height: 16),
              Text(
                'Checking subscription status...',
                style: TextStyle(
                  fontSize: 16,
                  color: isDarkMode ? Colors.white70 : Colors.black54,
                ),
                textAlign: TextAlign.center,
              ),
            ],
          ),
        ),
      );
    }

    if (subscriptionProvider.error != null) {
      final error = subscriptionProvider.error!;
      if (error.contains('Please log in') ||
          error.contains('Session expired')) {
        return Scaffold(
          backgroundColor: isDarkMode ? Colors.grey[900] : Colors.grey[50],
          body: Center(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Icon(
                  Icons.login_outlined,
                  size: 48,
                  color: Theme.of(context).primaryColor,
                ),
                const SizedBox(height: 16),
                Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 32),
                  child: Text(
                    'Authentication Required',
                    style: TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                      color: isDarkMode ? Colors.white : Colors.black87,
                    ),
                    textAlign: TextAlign.center,
                  ),
                ),
                const SizedBox(height: 8),
                Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 32),
                  child: Text(
                    'Please log in to access this feature',
                    style: TextStyle(
                      fontSize: 14,
                      color: isDarkMode ? Colors.white70 : Colors.black54,
                    ),
                    textAlign: TextAlign.center,
                  ),
                ),
                const SizedBox(height: 24),
                ElevatedButton.icon(
                  onPressed: () => subscriptionProvider.checkSubscription(),
                  icon: const Icon(Icons.refresh),
                  label: const Text('Retry'),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Theme.of(context).primaryColor,
                    foregroundColor: Colors.white,
                    padding: const EdgeInsets.symmetric(
                      horizontal: 24,
                      vertical: 12,
                    ),
                  ),
                ),
              ],
            ),
          ),
        );
      }
      return Scaffold(
        backgroundColor: isDarkMode ? Colors.grey[900] : Colors.grey[50],
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(
                Icons.error_outline,
                size: 48,
                color: Colors.red[400],
              ),
              const SizedBox(height: 16),
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 32),
                child: Text(
                  'Unable to verify subscription status',
                  style: TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                    color: isDarkMode ? Colors.white : Colors.black87,
                  ),
                  textAlign: TextAlign.center,
                ),
              ),
              const SizedBox(height: 8),
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 32),
                child: Text(
                  error,
                  style: TextStyle(
                    fontSize: 14,
                    color: isDarkMode ? Colors.white70 : Colors.black54,
                  ),
                  textAlign: TextAlign.center,
                ),
              ),
              const SizedBox(height: 24),
              ElevatedButton.icon(
                onPressed: () => subscriptionProvider.checkSubscription(),
                icon: const Icon(Icons.refresh),
                label: const Text('Retry'),
                style: ElevatedButton.styleFrom(
                  backgroundColor: Theme.of(context).primaryColor,
                  foregroundColor: Colors.white,
                  padding: const EdgeInsets.symmetric(
                    horizontal: 24,
                    vertical: 12,
                  ),
                ),
              ),
            ],
          ),
        ),
      );
    }

    // Safely get the current route name
    final currentRoute = ModalRoute.of(context);
    final routeName = currentRoute?.settings.name;
    final isSubscriptionPage = routeName == '/subscription_plans';

    // If we have premium access and we're on the subscription page, redirect to home
    if (_hasAccess(subscriptionProvider) && isSubscriptionPage) {
      WidgetsBinding.instance.addPostFrameCallback((_) {
        if (mounted && context.mounted) {
          Navigator.of(context).pushNamedAndRemoveUntil(
            '/',
            (Route<dynamic> route) => false,
          );
        }
      });
      return const SizedBox(); // Return empty widget while redirecting
    }

    // If we need premium access and don't have it
    if (widget.requirePremium && !_hasAccess(subscriptionProvider)) {
      // Only redirect if we're not already on the subscription page
      if (!isSubscriptionPage) {
        WidgetsBinding.instance.addPostFrameCallback((_) {
          if (mounted && context.mounted) {
            Navigator.of(context).pushNamedAndRemoveUntil(
              '/subscription_plans',
              (route) => false,
            );
          }
        });
        return const SizedBox(); // Return empty widget while redirecting
      }
    }

    return widget.child;
  }
}
