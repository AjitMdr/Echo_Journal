import 'subscription.dart';

class Payment {
  final int id;
  final int subscriptionId;
  final Subscription? subscriptionDetails;
  final double amount;
  final String currency;
  final String paymentMethod;
  final String status;
  final String transactionId;
  final DateTime paymentDate;
  final DateTime lastModified;

  Payment({
    required this.id,
    required this.subscriptionId,
    this.subscriptionDetails,
    required this.amount,
    required this.currency,
    required this.paymentMethod,
    required this.status,
    required this.transactionId,
    required this.paymentDate,
    required this.lastModified,
  });

  factory Payment.fromJson(Map<String, dynamic> json) {
    var rawAmount = json['amount'];
    double amount;

    if (rawAmount is String) {
      amount =
          double.tryParse(rawAmount.replaceAll(RegExp(r'[^\d.]'), '')) ?? 0.0;
    } else if (rawAmount is int) {
      amount = rawAmount.toDouble();
    } else if (rawAmount is double) {
      amount = rawAmount;
    } else {
      amount = 0.0;
    }

    return Payment(
      id: json['id'],
      subscriptionId: json['subscription'],
      subscriptionDetails: json['subscription_details'] != null
          ? Subscription.fromJson(json['subscription_details'])
          : null,
      amount: amount,
      currency: json['currency'],
      paymentMethod: json['payment_method'],
      status: json['status'],
      transactionId: json['transaction_id'],
      paymentDate: DateTime.parse(json['payment_date']),
      lastModified: DateTime.parse(json['last_modified']),
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'subscription': subscriptionId,
      'subscription_details': subscriptionDetails?.toJson(),
      'amount': amount,
      'currency': currency,
      'payment_method': paymentMethod,
      'status': status,
      'transaction_id': transactionId,
      'payment_date': paymentDate.toIso8601String(),
      'last_modified': lastModified.toIso8601String(),
    };
  }

  bool get isSuccessful => status == 'SUCCESS';
}
