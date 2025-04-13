import 'package:flutter/material.dart';
import '../../models/subscription/plan.dart';
import '../../models/subscription/subscription.dart';
import '../../services/subscription/subscription_service.dart';

class SubscriptionPage extends StatefulWidget {
  const SubscriptionPage({Key? key}) : super(key: key);

  @override
  _SubscriptionPageState createState() => _SubscriptionPageState();
}

class _SubscriptionPageState extends State<SubscriptionPage> {
  final SubscriptionService _subscriptionService = SubscriptionService();
  Subscription? _currentSubscription;
  List<Plan> _plans = [];
  bool _isLoading = true;
  String? _error;

  @override
  void initState() {
    super.initState();
    _loadData();
  }

  Future<void> _loadData() async {
    setState(() {
      _isLoading = true;
      _error = null;
    });

    try {
      final plans = await _subscriptionService.getPlans();
      try {
        final subscription =
            await _subscriptionService.getCurrentSubscription();
        setState(() {
          _currentSubscription = subscription;
        });
      } catch (e) {
        // No active subscription, that's okay
      }

      setState(() {
        _plans = plans;
        _isLoading = false;
      });
    } catch (e) {
      setState(() {
        _error = e.toString();
        _isLoading = false;
      });
    }
  }

  Future<void> _subscribe(Plan plan) async {
    try {
      setState(() => _isLoading = true);
      final subscription = await _subscriptionService.subscribe(plan.id);
      setState(() {
        _currentSubscription = subscription;
        _isLoading = false;
      });
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Successfully subscribed to ${plan.name}')),
        );
      }
    } catch (e) {
      setState(() => _isLoading = false);
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Failed to subscribe: ${e.toString()}')),
        );
      }
    }
  }

  Future<void> _cancelSubscription() async {
    if (_currentSubscription == null) return;

    try {
      setState(() => _isLoading = true);
      await _subscriptionService.cancelSubscription();
      await _loadData(); // Reload data to get updated subscription status
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Subscription cancelled successfully')),
        );
      }
    } catch (e) {
      setState(() => _isLoading = false);
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Failed to cancel subscription: ${e.toString()}'),
          ),
        );
      }
    }
  }

  Widget _buildPlanCard(Plan plan) {
    final isCurrentPlan = _currentSubscription?.planId == plan.id;
    final isActive = _currentSubscription?.isActive ?? false;

    return Card(
      elevation: 4,
      margin: const EdgeInsets.all(8),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(plan.name, style: Theme.of(context).textTheme.titleLarge),
            const SizedBox(height: 8),
            Text(
              '\Rs ${plan.price.toStringAsFixed(2)} / ${plan.durationDays} days',
              style: Theme.of(context).textTheme.titleMedium,
            ),
            const SizedBox(height: 8),
            Text(plan.description),
            const SizedBox(height: 16),
            if (plan.features.isNotEmpty) ...[
              Text('Features:', style: Theme.of(context).textTheme.titleMedium),
              const SizedBox(height: 8),
              ...plan.features.entries.map(
                (feature) => Padding(
                  padding: const EdgeInsets.only(left: 16, bottom: 4),
                  child: Row(
                    children: [
                      const Icon(Icons.check, size: 16),
                      const SizedBox(width: 8),
                      Expanded(child: Text(feature.value.toString())),
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 16),
            ],
            if (isCurrentPlan && isActive)
              ElevatedButton(
                onPressed: _cancelSubscription,
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.red,
                  minimumSize: const Size(double.infinity, 36),
                ),
                child: const Text('Cancel Subscription'),
              )
            else
              ElevatedButton(
                onPressed: isActive ? null : () => _subscribe(plan),
                style: ElevatedButton.styleFrom(
                  minimumSize: const Size(double.infinity, 36),
                ),
                child: Text(isActive ? 'Already Subscribed' : 'Subscribe'),
              ),
          ],
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Subscription Plans')),
      body: _isLoading
          ? const Center(child: CircularProgressIndicator())
          : _error != null
              ? Center(
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Text('Error: $_error'),
                      const SizedBox(height: 16),
                      ElevatedButton(
                        onPressed: _loadData,
                        child: const Text('Retry'),
                      ),
                    ],
                  ),
                )
              : RefreshIndicator(
                  onRefresh: _loadData,
                  child: ListView(
                    padding: const EdgeInsets.all(16),
                    children: [
                      if (_currentSubscription != null) ...[
                        Card(
                          child: Padding(
                            padding: const EdgeInsets.all(16),
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Text(
                                  'Current Subscription',
                                  style: Theme.of(context).textTheme.titleLarge,
                                ),
                                const SizedBox(height: 8),
                                Text(
                                  'Status: ${_currentSubscription!.status}',
                                  style: TextStyle(
                                    color: _currentSubscription!.isActive
                                        ? Colors.green
                                        : Colors.red,
                                  ),
                                ),
                                const SizedBox(height: 4),
                                Text(
                                  'Days Remaining: ${_currentSubscription!.daysRemaining}',
                                ),
                                if (_currentSubscription!.isAutoRenewal)
                                  const Text('Auto-renewal: Enabled'),
                              ],
                            ),
                          ),
                        ),
                        const SizedBox(height: 24),
                        Text(
                          'Available Plans',
                          style: Theme.of(context).textTheme.titleLarge,
                        ),
                        const SizedBox(height: 16),
                      ],
                      ..._plans.map(_buildPlanCard),
                    ],
                  ),
                ),
    );
  }
}
