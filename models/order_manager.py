import os
import pandas as pd
from datetime import datetime
from ..utils.logger import logger, log_order
from ..config import settings

class OrderManager:
    def __init__(self, broker):
        """Initialize the OrderManager.

        Args:
            broker: Broker instance for placing orders
        """
        self.broker = broker
        self.last_order_id = None
        self.order_history = []
        self.order_counter = 0
        
        # Ensure order history directory exists
        os.makedirs(settings.ORDER_HISTORY_DIR, exist_ok=True)

    def place_order(self, signal_data):
        """Place an order based on signal data.

        Args:
            signal_data (dict): Signal data with order parameters

        Returns:
            str: Order ID if successful, None otherwise
        """
        if not self.broker:
            logger.error("Broker not available for placing order")
            return None

        if not signal_data:
            logger.error("Invalid signal data for order placement")
            return None

        # Check if we already have reached maximum orders for today
        if self.order_counter >= settings.MAX_SIGNAL_ATTEMPTS:
            logger.info(f"Maximum signal attempts reached for today ({self.order_counter}). Skipping.")
            return None

        # Prepare order parameters for bracket order
        order_params = {
            "symboltoken": signal_data.get('token'),
            "symbol": signal_data.get('symbol'),
            "quantity": signal_data.get('quantity', 1),
            "ordertype": "MARKET",
            "tradingsymbol": signal_data.get('trading_symbol'),
            "producttype": signal_data.get('product_type', 'INTRADAY'),
            "duration": signal_data.get('duration', 'DAY'),
            "price": signal_data.get('price', 0),
            "squareoff": signal_data.get('target', 0),
            "stoploss": signal_data.get('stop_loss', 0),
            "transactiontype": signal_data.get('transaction_type', 'BUY')
        }

        # Place bracket order
        order_id = self.broker.place_order(order_params, order_type="BO")

        if order_id:
            # Increment order counter
            self.order_counter += 1

            # Record order details
            order_record = {
                'order_id': order_id,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'symbol': signal_data.get('symbol'),
                'token': signal_data.get('token'),
                'quantity': signal_data.get('quantity'),
                'price': signal_data.get('price'),
                'stop_loss': signal_data.get('stop_loss'),
                'target': signal_data.get('target'),
                'transaction_type': signal_data.get('transaction_type'),
                'product_type': signal_data.get('product_type'),
                'order_type': 'BO',
                'status': 'PLACED'
            }

            # Save order record
            self._save_order_to_csv(order_record)
            self.order_history.append(order_record)
            self.last_order_id = order_id

            logger.info(f"Bracket order placed successfully with ID: {order_id}")
            return order_id

        logger.error("Failed to place order")
        return None

    def check_order_status(self, order_id):
        """Check the status of a placed order.

        Args:
            order_id (str): Order ID to check

        Returns:
            dict: Order status information or None if failed
        """
        return self.broker.check_order_status(order_id)

    def _save_order_to_csv(self, order_data):
        """Save order data to a CSV file.

        Args:
            order_data (dict): Order details
        """
        try:
            date_str = datetime.now().strftime('%Y%m%d')
            filename = os.path.join(settings.ORDER_HISTORY_DIR, f'order_history_{date_str}.csv')

            # Convert all values to string to avoid type conflicts
            for key, value in order_data.items():
                order_data[key] = str(value)

            # Create DataFrame and save to CSV
            df = pd.DataFrame([order_data])
            if os.path.exists(filename):
                df.to_csv(filename, mode='a', header=False, index=False)
            else:
                df.to_csv(filename, index=False)

            logger.info(f"Saved order details to {filename}")
        except Exception as e:
            logger.error(f"Error saving order to CSV: {e}")