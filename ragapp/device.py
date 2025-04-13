from enum import Enum


class DeviceEnum(str, Enum):
    lifepak_15 = "LIFEPAK 15"
    lifepak_20 = "LIFEPAK 20"
    mizuho_6800 = "MIZUHO 6800"
    philips_m3002 = "PHILIPS M3002"
    defib_misc = "DEFIB MISC"
    philips_m3015 = "PHILIPS M3015"
    philips_m4841 = "PHILIPS M4841"
    philips_v60_vent = "PHILIPS V60 VENT"


LP_DEVICES = (DeviceEnum.lifepak_15, DeviceEnum.lifepak_20)
