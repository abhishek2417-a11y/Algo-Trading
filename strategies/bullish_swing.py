"""
Bullish Swing Strategy module.
Implements a trading strategy based on bullish swing patterns.
"""
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime

from ..utils.logger import logger
from ..models.option import OptionData
from ..models.order_manager import OrderManager
from ..config import settings

class BullishSwingStrategy:
    """
    Strategy that identifies and trades bullish swing patterns.
    The strategy looks for specific price patterns (L1, H1, A, B, C, D points)
    and generates trading signals when price breaks above the D point.
    """
    
    def __init__(self, data, order_manager=None):
        """
        Initialize the strategy with price data and order manager.
        
        Args:
            data (pd.DataFrame): Price data with OHLCV columns
            order_manager: OrderManager instance for handling orders
        """
        # Store the data and order manager
        self.data = data.copy() if not data.empty else pd.DataFrame()
        self.order_manager = order_manager
        
        # Initialize signal tracking
        self.signals = pd.DataFrame(index=self.data.index) if not self.data.empty else pd.DataFrame()
        self.signals['signal'] = 0
        self.signals['entry_price'] = np.nan
        self.signals['stop_loss'] = np.nan
        self.signals['target'] = np.nan
        
        # Initialize structure points
        self.H1 = None
        self.L1 = None
        self.A = None
        self.B = None
        self.C = None
        self.D = None
        
        # Structure point indices
        self.H1_idx = None
        self.L1_idx = None
        self.A_idx = None
        self.B_idx = None
        self.C_idx = None
        self.D_idx = None
        
        # Setup and order tracking
        self.pending_setup = None
        self.structures = []
        self.executed_patterns = []
        self.options_data = None
        self.signal_counter = 0
        
        # Initialize order manager
        self.order_manager = OrderManager(broker) if broker else None
        
        # Greeks data caching
        self.last_greeks_refresh = None
        self.greeks_refresh_interval = settings.GREEKS_REFRESH_INTERVAL
        self.cached_options_data = None
        
        # Create directories for storing data
        os.makedirs(settings.ORDER_HISTORY_DIR, exist_ok=True)
        os.makedirs(settings.OPTIONS_DATA_DIR, exist_ok=True)
    
    def _check_for_entry_trigger(self, idx):
        """
        Check if entry conditions are met at the given index.
        
        Args:
            idx (int): Index in the data
            
        Returns:
            bool: True if entry conditions are met, False otherwise
        """
        if self.signals.iloc[idx]['signal'] == 1:
            return True
        return False
    
    def print_current_structure(self):
        """Print the current bullish structure information."""
        logger.info("\n===== CURRENT BULLISH STRUCTURE =====")
        
        def format_point(name, value, idx):
            if value is not None and idx is not None:
                time_str = self.data.index[idx]
                return f"{name}: {value:.2f} at {time_str}"
            return f"{name}: Not yet formed"
        
        logger.info(format_point("L1", self.L1, self.L1_idx))
        logger.info(format_point("H1", self.H1, self.H1_idx))
        logger.info(format_point("A", self.A, self.A_idx))
        
        # B is a swing low point in the bullish structure
        if self.B is not None and self.B_idx is not None:
            time_str = self.data.index[self.B_idx]
            logger.info(f"B (swing low): {self.B:.2f} at {time_str}")
        else:
            logger.info("B (swing low): Not yet formed")
            
        logger.info(format_point("C", self.C, self.C_idx))
        logger.info(format_point("D", self.D, self.D_idx))
        
        if self.pending_setup:
            logger.info("\n----- PENDING SETUP -----")
            logger.info(f"Entry Price: {self.pending_setup['entry_price']:.2f}")
            logger.info(f"Stop Loss: {self.pending_setup['stop_loss']:.2f}")
            logger.info(f"Target: {self.pending_setup['target']:.2f}")
            
            # Display enhanced risk management information
            risk = self.pending_setup.get('risk_per_unit', self.pending_setup['entry_price'] - self.pending_setup['stop_loss'])
            reward = self.pending_setup['target'] - self.pending_setup['entry_price']
            risk_reward = self.pending_setup.get('risk_reward_ratio', reward / risk if risk > 0 else 0)
            
            logger.info(f"Risk:Reward - 1:{risk_reward:.2f}")
            logger.info(f"Position Size: {self.pending_setup.get('position_size', settings.DEFAULT_LOT_SIZE)} units")
            logger.info(f"Max Risk Amount: {self.pending_setup.get('max_risk_amount', 0):.2f}")
            
            # Calculate and display total risk for the trade
            total_risk = risk * self.pending_setup.get('position_size', settings.DEFAULT_LOT_SIZE)
            logger.info(f"Total Risk: {total_risk:.2f}")
            
            # Display volatility information
            volatility_factor = self._calculate_volatility_factor()
            logger.info(f"Market Volatility Factor: {volatility_factor:.2f}")
            
        logger.info("====================================\n")
    
    def calculate_point_D(self):
        """
        Calculate the D point of the bullish structure.
        This is a critical point for signal generation.
        """
        if self.H1 is not None and self.B is not None and self.C is not None:
            logger.info(f"Calculating D point with B={self.B:.2f} (swing low), C={self.C:.2f}")
            
            # Initialize D with B's price (B is a swing low, and D should be a high after B)
            highest_high = self.data.iloc[self.B_idx]['high']
            highest_high_idx = self.B_idx
            
            # Scan from B to C (inclusive) to find the highest high
            for i in range(self.B_idx + 1, self.C_idx + 1):
                current_high = self.data.iloc[i]['high']
                if current_high > highest_high:
                    highest_high = current_high
                    highest_high_idx = i
                    
            self.D = highest_high
            self.D_idx = highest_high_idx
            d_time = self.data.index[self.D_idx]
            
            logger.info(f"POINT CALCULATION: D calculated at price {self.D:.2f} at time {d_time}")
            logger.info("\n===== COMPLETE STRUCTURE DETAILS =====")
            logger.info(f"L1: {self.L1:.2f} at {self.data.index[self.L1_idx]}")
            logger.info(f"H1: {self.H1:.2f} at {self.data.index[self.H1_idx]}")
            logger.info(f"A: {self.A:.2f} at {self.data.index[self.A_idx]}")
            logger.info(f"B: {self.B:.2f} (swing low) at {self.data.index[self.B_idx]}")
            logger.info(f"C: {self.C:.2f} at {self.data.index[self.C_idx]}")
            logger.info(f"D: {self.D:.2f} at {self.data.index[self.D_idx]}")
            logger.info("======================================\n")
            
            # Calculate trade parameters with enhanced risk management
            entry_price = self.D + 0.05  # Small buffer above D
            stop_loss = self.C - 0.05  # Below C point for stop loss
            risk = entry_price - stop_loss
            
            # Dynamic risk-reward ratio based on market conditions
            # Use higher RR when market is trending strongly, lower when choppy
            volatility_factor = self._calculate_volatility_factor()
            risk_reward_ratio = max(2.0, 2.0 + volatility_factor)  # Minimum 1:2, can increase based on volatility
            target = entry_price + (risk * risk_reward_ratio)
            
            # Calculate position size based on risk parameters
            max_risk_amount = self._calculate_max_risk_amount()
            position_size = self._calculate_position_size(risk, max_risk_amount)
            
            self.pending_setup = {
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'target': target,
                'risk_reward_ratio': risk_reward_ratio,
                'position_size': position_size,
                'max_risk_amount': max_risk_amount,
                'risk_per_unit': risk,
                'structure': {
                    'H1': (self.H1_idx, self.H1),
                    'L1': (self.L1_idx, self.L1),
                    'A': (self.A_idx, self.A),
                    'B': (self.B_idx, self.B),
                    'C': (self.C_idx, self.C),
                    'D': (self.D_idx, self.D)
                },
                'options_data': None,
                'strategy_type': 'CE_BUYING'
            }
            
            # Fetch options data if broker is available
            if self.broker and hasattr(self.broker, 'api'):
                logger.info("Fetching option Greeks for CE buying setup...")
                self.refresh_option_greeks(current_idx=self.D_idx, force_refresh=True)
    
    def is_swing_high(self, idx):
        """
        Check if the given index represents a swing high.
        
        Args:
            idx (int): Index in the data to check
            
        Returns:
            bool: True if it's a swing high, False otherwise
        """
        if idx <= 0 or idx >= len(self.data) - 1:
            return False
            
        current_high = self.data.iloc[idx]['high']
        prev_high = self.data.iloc[idx-1]['high']
        next_high = self.data.iloc[idx+1]['high']
        
        return current_high > prev_high and current_high > next_high
    
    def is_swing_low(self, idx):
        """
        Check if the given index represents a swing low.
        
        Args:
            idx (int): Index in the data to check
            
        Returns:
            bool: True if it's a swing low, False otherwise
        """
        if idx <= 0 or idx >= len(self.data) - 1:
            return False
            
        current_low = self.data.iloc[idx]['low']
        prev_low = self.data.iloc[idx-1]['low']
        next_low = self.data.iloc[idx+1]['low']
        
        return current_low < prev_low and current_low < next_low
    
    def detect_swings(self):
        """Detect swing highs and lows in the data."""
        self.swing_highs = []
        self.swing_lows = []
        
        for i in range(1, len(self.data) - 1):
            if self.is_swing_high(i):
                self.swing_highs.append((i, self.data.iloc[i]['high']))
                
            if self.is_swing_low(i):
                self.swing_lows.append((i, self.data.iloc[i]['low']))
    
    def initialize_H1_L1(self):
        """Initialize L1 using the low of the first candle."""
        if len(self.data) > 0:
            self.L1 = self.data.iloc[0]['low']
            self.L1_idx = 0
            logger.info(f"POINT INITIALIZATION: L1 initialized to first candle low at {self.L1:.2f}")
            
            # Reset H1 to None - it will be identified later
            self.H1 = None
            self.H1_idx = None
        else:
            logger.warning("No data available to initialize L1 point")
    
    def reset_points(self, which_points):
        """
        Reset specified structure points.
        
        Args:
            which_points (list): List of point names to reset (e.g., ['H1', 'L1'])
        """
        for point in which_points:
            if point == 'H1':
                self.H1 = None
                self.H1_idx = None
            elif point == 'L1':
                self.L1 = None
                self.L1_idx = None
            elif point == 'A':
                self.A = None
                self.A_idx = None
            elif point == 'B':
                self.B = None
                self.B_idx = None
            elif point == 'C':
                self.C = None
                self.C_idx = None
            elif point == 'D':
                self.D = None
                self.D_idx = None
                self.pending_setup = None
    
    def refresh_option_greeks(self, current_idx=None, force_refresh=False, min_refresh_interval=600, display_mapping=True):
        """
        Refresh option Greeks data and select appropriate options for trading.
        
        Args:
            current_idx (int, optional): Current index in the data
            force_refresh (bool): Force refresh regardless of time interval
            min_refresh_interval (int): Minimum time between refreshes in seconds
            display_mapping (bool): Whether to display option mapping details
            
        Returns:
            bool: True if refresh was successful, False otherwise
        """
        current_time = time.time()
        
        # Check if we need to refresh based on time interval
        if not force_refresh and self.last_greeks_refresh and \
           (current_time - self.last_greeks_refresh) < min_refresh_interval:
            logger.info(f"Skipping Greeks refresh - last refresh was {current_time - self.last_greeks_refresh:.1f}s ago")
            return False
            
        if not self.broker or not hasattr(self.broker, 'api'):
            logger.warning("No broker API available for fetching option data")
            return False
            
        try:
            # Use current index if provided, otherwise use the latest data point
            if current_idx is None:
                current_idx = len(self.data) - 1
                
            current_price = self.data.iloc[current_idx]['close']
            logger.info(f"Fetching option Greeks at price {current_price:.2f}")
            
            # This would be implemented in the broker to fetch options data
            # For now, we'll simulate it with a placeholder
            options_data = self._fetch_options_data(current_price)
            
            if options_data is None or options_data.empty:
                logger.warning("Failed to fetch options data")
                return False
                
            self.options_data = options_data
            self.last_greeks_refresh = current_time
            
            # Process the options data for CE buying strategy
            self._process_options_for_strategy(current_price, display_mapping)
            
            return True
            
        except Exception as e:
            logger.error(f"Error refreshing option Greeks: {e}")
            return False
            
    def _fetch_options_data(self, current_price):
        """
        Fetch options data from the broker.
        This is a placeholder that would be implemented with actual broker API calls.
        
        Args:
            current_price (float): Current underlying price
            
        Returns:
            pd.DataFrame: Options data or None if failed
        """
        try:
            # This would be implemented with actual broker API calls
            # For now, return cached data if available
            if self.cached_options_data is not None:
                return self.cached_options_data
                
            # Placeholder for actual implementation
            logger.info("This is a placeholder for actual options data fetching")
            return None
            
        except Exception as e:
            logger.error(f"Error fetching options data: {e}")
            return None
            
    def _calculate_volatility_factor(self):
        """
        Calculate a volatility factor based on recent price action.
        Higher volatility may warrant higher risk-reward ratios.
        
        Returns:
            float: Volatility factor (0.0 to 1.0)
        """
        try:
            # Use the last 20 candles to calculate volatility
            lookback = min(20, len(self.data) - 1)
            if lookback <= 0:
                return 0.0
                
            # Calculate the average true range (ATR) as a volatility measure
            high_low = self.data['high'][-lookback:] - self.data['low'][-lookback:]
            high_close = abs(self.data['high'][-lookback:] - self.data['close'][-lookback-1:-1].values)
            low_close = abs(self.data['low'][-lookback:] - self.data['close'][-lookback-1:-1].values)
            
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            atr = true_range.mean()
            
            # Calculate the average price over the same period
            avg_price = self.data['close'][-lookback:].mean()
            
            # Normalize ATR as a percentage of average price
            normalized_atr = atr / avg_price if avg_price > 0 else 0
            
            # Map normalized ATR to a volatility factor (0.0 to 1.0)
            # Higher ATR = higher volatility = higher factor
            volatility_factor = min(1.0, normalized_atr * 100)  # Scale appropriately
            
            logger.info(f"Calculated volatility factor: {volatility_factor:.2f} (ATR: {atr:.2f}, Avg Price: {avg_price:.2f})")
            return volatility_factor
            
        except Exception as e:
            logger.error(f"Error calculating volatility factor: {e}")
            return 0.0
    
    def _calculate_max_risk_amount(self):
        """
        Calculate the maximum risk amount for this trade based on account settings.
        
        Returns:
            float: Maximum risk amount in currency units
        """
        try:
            # Get account balance from broker if available
            account_balance = 0
            if self.broker and hasattr(self.broker, 'get_account_balance'):
                account_balance = self.broker.get_account_balance()
            else:
                # Use default value from settings if broker not available
                account_balance = settings.DEFAULT_ACCOUNT_BALANCE
            
            # Calculate risk based on risk percentage from settings
            risk_percentage = settings.TARGET_RISK_MID / 100.0  # Convert percentage to decimal
            max_risk = account_balance * risk_percentage
            
            # Apply additional risk adjustments based on market conditions
            # For example, reduce risk in high volatility environments
            volatility_factor = self._calculate_volatility_factor()
            if volatility_factor > 0.5:  # High volatility
                max_risk *= (1.0 - (volatility_factor - 0.5))  # Reduce risk as volatility increases
            
            logger.info(f"Maximum risk amount: {max_risk:.2f} (Account: {account_balance:.2f}, Risk %: {risk_percentage:.2%})")
            return max_risk
            
        except Exception as e:
            logger.error(f"Error calculating max risk amount: {e}")
            return settings.DEFAULT_MAX_RISK
    
    def _calculate_position_size(self, risk_per_unit, max_risk_amount):
        """
        Calculate the appropriate position size based on risk parameters.
        
        Args:
            risk_per_unit (float): Risk per unit (entry price - stop loss)
            max_risk_amount (float): Maximum risk amount in currency units
            
        Returns:
            int: Position size in number of units/lots
        """
        try:
            if risk_per_unit <= 0:
                logger.warning("Invalid risk per unit (must be > 0)")
                return settings.DEFAULT_LOT_SIZE
                
            # Calculate raw position size based on risk
            raw_position_size = max_risk_amount / risk_per_unit
            
            # Round down to nearest lot size
            lot_size = settings.DEFAULT_LOT_SIZE
            position_size = int(raw_position_size / lot_size) * lot_size
            
            # Ensure position size is within limits
            min_position = settings.MIN_POSITION_SIZE if hasattr(settings, 'MIN_POSITION_SIZE') else lot_size
            max_position = settings.MAX_POSITION_SIZE if hasattr(settings, 'MAX_POSITION_SIZE') else lot_size * 10
            
            position_size = max(min_position, min(position_size, max_position))
            
            logger.info(f"Calculated position size: {position_size} units (Risk/Unit: {risk_per_unit:.2f}, Max Risk: {max_risk_amount:.2f})")
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return settings.DEFAULT_LOT_SIZE
    
    def _process_options_for_strategy(self, current_price, display_mapping=True):
        """
        Process options data for the current strategy.
        Selects appropriate options based on strategy type and current price.
        
        Args:
            current_price (float): Current underlying price
            display_mapping (bool): Whether to display option mapping details
        """
        if self.options_data is None or self.options_data.empty:
            logger.warning("No options data available for processing")
            return
            
        try:
            # Filter for CE options for bullish strategy
            ce_options = self.options_data[self.options_data['option_type'] == 'CE'].copy()
            if ce_options.empty:
                logger.warning("No CE options found in options data")
                return
                
            # Get unique expiry dates and sort them
            expiry_dates = sorted(ce_options['expiry_date'].unique())
            if len(expiry_dates) < 2:
                logger.warning(f"Insufficient expiry dates found: {len(expiry_dates)}")
                return
                
            current_expiry = expiry_dates[0]
            next_expiry = expiry_dates[1]
            
            logger.info(f"Processing options for current expiry: {current_expiry} and next expiry: {next_expiry}")
            
            # Process current expiry options
            current_exp_options = ce_options[ce_options['expiry_date'] == current_expiry].copy()
            current_exp_options['distance_from_price'] = abs(current_exp_options['strike_float'] - current_price)
            current_exp_options = current_exp_options.sort_values('distance_from_price')
            
            # Find true ATM strike (closest strike below or equal to current price)
            atm_options = current_exp_options[current_exp_options['strike_float'] <= current_price]
            if not atm_options.empty:
                atm_strike = atm_options.iloc[0]['strike_float']
            else:
                # If no strikes below current price, take the lowest OTM strike
                atm_strike = current_exp_options.iloc[0]['strike_float']
                
            # Select strikes for current expiry
            current_exp_selected = pd.DataFrame()
            
            # 1 nearest OTM (strictly greater than current price)
            otm_options = current_exp_options[current_exp_options['strike_float'] > current_price].sort_values('strike_float')
            if not otm_options.empty:
                current_exp_selected = pd.concat([current_exp_selected, otm_options.head(1)])
                
            # 1 ATM
            atm_option = current_exp_options[current_exp_options['strike_float'] == atm_strike]
            if not atm_option.empty:
                current_exp_selected = pd.concat([current_exp_selected, atm_option]).drop_duplicates()
                
            # 4 nearest ITM
            itm_options = current_exp_options[current_exp_options['strike_float'] < current_price].sort_values('strike_float', ascending=False)
            if not itm_options.empty:
                current_exp_selected = pd.concat([current_exp_selected, itm_options.head(4)]).drop_duplicates()
                
            # Process next expiry options
            next_exp_options = ce_options[ce_options['expiry_date'] == next_expiry].copy()
            next_exp_options['distance_from_price'] = abs(next_exp_options['strike_float'] - current_price)
            next_exp_options = next_exp_options.sort_values('distance_from_price')
            
            # Find true ATM strike for next expiry
            next_atm_options = next_exp_options[next_exp_options['strike_float'] <= current_price]
            if not next_atm_options.empty:
                next_atm_strike = next_atm_options.iloc[0]['strike_float']
            else:
                next_atm_strike = next_exp_options.iloc[0]['strike_float']
                
            # Select strikes for next expiry
            next_exp_selected = pd.DataFrame()
            
            # 1 nearest OTM
            next_otm_options = next_exp_options[next_exp_options['strike_float'] > current_price].sort_values('strike_float')
            if not next_otm_options.empty:
                next_exp_selected = pd.concat([next_exp_selected, next_otm_options.head(1)]).drop_duplicates()
                
            # 1 ATM
            next_atm_option = next_exp_options[next_exp_options['strike_float'] == next_atm_strike]
            if not next_atm_option.empty:
                next_exp_selected = pd.concat([next_exp_selected, next_atm_option]).drop_duplicates()
                
            # 3 nearest ITM
            next_itm_options = next_exp_options[next_exp_options['strike_float'] < current_price].sort_values('strike_float', ascending=False)
            if not next_itm_options.empty:
                next_exp_selected = pd.concat([next_exp_selected, next_itm_options.head(3)]).drop_duplicates()
                
            # Combine selected options from both expiries
            selected_options = pd.concat([current_exp_selected, next_exp_selected])
            selected_options = selected_options.drop_duplicates()
            
            # Store the selected options for trading
            if not selected_options.empty:
                self.pending_setup['options_data'] = selected_options
                logger.info(f"Selected {len(selected_options)} options for trading")
                
                if display_mapping:
                    logger.info("\n===== SELECTED OPTIONS =====")
                    for _, option in selected_options.iterrows():
                        logger.info(f"Strike: {option['strike_float']}, Type: {option['option_type']}, "  
                                   f"Expiry: {option['expiry_date']}, Delta: {option.get('delta', 'N/A')}")
                    logger.info("============================\n")
            else:
                logger.warning("No suitable options selected for trading")
                
        except Exception as e:
            logger.error(f"Error processing options for strategy: {e}")

    
    def calculate_option_stop_loss(self, option_data, underlying_entry, underlying_sl):
        """
        Calculate option-specific stop loss using Greeks and underlying movement.
        
        Args:
            option_data (dict): Option data with Greeks
            underlying_entry (float): Entry price of the underlying
            underlying_sl (float): Stop loss price of the underlying
            
        Returns:
            dict: Stop loss calculation details
        """
        try:
            # Convert option_data to OptionData object if it's a dict
            if isinstance(option_data, dict):
                option = OptionData(option_data)
            else:
                option = option_data
                
            return option.calculate_stop_loss(underlying_entry, underlying_sl)
                
        except Exception as e:
            logger.error(f"Error calculating option stop loss: {e}")
            return None
    
    def check_live_tick(self, price):
        """
        Check for breakout signals in live ticks.
        
        Args:
            price (float): Current price from the tick
            
        Returns:
            dict: Signal data if breakout detected, None otherwise
        """
        if self.C is not None and self.D is not None:
            logger.info(f"Checking live tick at price {price:.2f} against D point {self.D:.2f}")
            
            if price > self.D:
                # Check if we already have an order placed today
                if self.order_counter >= settings.MAX_SIGNAL_ATTEMPTS:
                    logger.info(f"Maximum signal attempts reached for today ({self.order_counter}). Skipping.")
                    return None
                    
                logger.info(f"LIVE BREAKOUT DETECTED: Price {price:.2f} broke above D point {self.D:.2f}")
                
                if self.pending_setup is None:
                    # Calculate setup parameters
                    entry_price = self.D + 0.05
                    stop_loss = self.C - 0.05
                    risk_points = entry_price - stop_loss
                    target = entry_price + (2 * risk_points)
                    
                    self.pending_setup = {
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'target': target,
                        'structure': {
                            'H1': (self.H1_idx, self.H1),
                            'L1': (self.L1_idx, self.L1),
                            'A': (self.A_idx, self.A),
                            'B': (self.B_idx, self.B),
                            'C': (self.C_idx, self.C),
                            'D': (self.D_idx, self.D)
                        }
                    }
                
                # Prepare options data for signal
                if self.cached_options_data is not None and not self.cached_options_data.empty:
                    options_data = self.cached_options_data
                else:
                    self.refresh_option_greeks(force_refresh=True)
                    options_data = self.cached_options_data
                
                if options_data is None or options_data.empty:
                    logger.error("No options data available for signal generation")
                    return None
                
                # Select the optimal option strike
                current_price = price
                entry_price = self.pending_setup['entry_price']
                stop_loss = self.pending_setup['stop_loss']
                
                # Use the OptionData static method to select the optimal strike
                option, quantity, total_risk = OptionData.select_optimal_strike(
                    options_data, 
                    current_price,
                    target_risk_range=(settings.TARGET_RISK_MIN, settings.TARGET_RISK_MAX),
                    lot_size=settings.DEFAULT_LOT_SIZE
                )
                
                if option is None:
                    logger.error("Failed to select an optimal option strike")
                    return None
                
                # Create signal data
                signal_data = {
                    'signal': 1,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'target': self.pending_setup['target'],
                    'option_data': option.to_dict(),
                    'quantity': quantity,
                    'total_risk': total_risk,
                    'timestamp': datetime.now()
                }
                
                self.signal_counter += 1
                return signal_data
                
        return None
    
    def place_order(self, signal_data):
        """
        Place an order based on signal data using the OrderManager.

        Args:
            signal_data (dict): Signal data with order parameters

        Returns:
            str: Order ID if successful, None otherwise
        """
        if not self.order_manager:
            logger.error("OrderManager not available for placing order")
            return None

        return self.order_manager.place_order(signal_data)
    
    
    def generate_signals(self):
        """
        Process historical data to identify structure points and generate signals.
        
        Returns:
            pd.DataFrame: DataFrame with signal information
        """
        # Initialize structure
        self.initialize_H1_L1()
        
        for i in range(1, len(self.data)):
            current_candle = self.data.iloc[i]
            prev_candle = self.data.iloc[i-1]
            current_time = self.data.index[i]
            
            # L1 detection and updates
            if self.is_swing_low(i) and current_candle['low'] < self.L1:
                previous_L1 = self.L1
                self.L1 = current_candle['low']
                self.L1_idx = i
                self.reset_points(['H1', 'A', 'B', 'C', 'D'])
                logger.info(f"POINT UPDATE: L1 updated from {previous_L1:.2f} to {self.L1:.2f}")
                
            # H1 detection and updates
            if self.L1 is not None and self.is_swing_high(i):
                if i > self.L1_idx and current_candle['high'] > self.L1:
                    if self.H1 is None:
                        self.H1 = current_candle['high']
                        self.H1_idx = i
                        logger.info(f"POINT DETECTION: H1 detected at price {self.H1:.2f}")
                    elif current_candle['high'] > self.H1:
                        previous_H1 = self.H1
                        self.H1 = current_candle['high']
                        self.H1_idx = i
                        self.reset_points(['A', 'B', 'C', 'D'])
                        logger.info(f"POINT UPDATE: H1 updated from {previous_H1:.2f} to {self.H1:.2f}")
                        
            # A point detection
            if self.H1 is not None and self.L1 is not None:
                if self.A is None and self.is_swing_low(i):
                    if i > self.H1_idx and current_candle['low'] > self.L1:
                        self.A = current_candle['low']
                        self.A_idx = i
                        logger.info(f"POINT DETECTION: A detected at price {self.A:.2f}")
                        
            # B point detection
            if self.A is not None:
                if self.B is None and self.is_swing_low(i):
                    if i > self.A_idx + 1 and current_candle['low'] > self.A:
                        self.B = current_candle['low']
                        self.B_idx = i
                        logger.info(f"POINT DETECTION: B detected at price {self.B:.2f}")
                        
            # C and D point detection
            if self.B is not None:
                if self.C is None:
                    if i > self.B_idx and current_candle['low'] < prev_candle['low'] and current_candle['low'] > self.B:
                        self.C = current_candle['low']
                        self.C_idx = i
                        logger.info(f"POINT DETECTION: C detected at price {self.C:.2f}")
                        self.calculate_point_D()
                        
            # Breakout detection
            if self.pending_setup is not None:
                if current_candle['high'] > self.D:
                    # Set signal in the signals DataFrame
                    self.signals.iloc[i, self.signals.columns.get_loc('signal')] = 1
                    self.signals.iloc[i, self.signals.columns.get_loc('entry_price')] = self.pending_setup['entry_price']
                    self.signals.iloc[i, self.signals.columns.get_loc('stop_loss')] = self.pending_setup['stop_loss']
                    self.signals.iloc[i, self.signals.columns.get_loc('target')] = self.pending_setup['target']
                    
                    # Record the structure
                    structure = self.pending_setup['structure'].copy()
                    structure['entry'] = (i, self.pending_setup['entry_price'])
                    structure['stop_loss'] = (i, self.pending_setup['stop_loss'])
                    structure['target'] = (i, self.pending_setup['target'])
                    self.structures.append(structure)
                    
                    logger.info(f"Signal generated at {self.data.index[i]}. Entry: {self.pending_setup['entry_price']:.2f}")
                    
                    # Reset points to start looking for new structure
                    self.reset_points(['H1', 'L1', 'A', 'B', 'C', 'D'])
                    self.L1 = current_candle['low']
                    self.L1_idx = i
        
        return self.signals
