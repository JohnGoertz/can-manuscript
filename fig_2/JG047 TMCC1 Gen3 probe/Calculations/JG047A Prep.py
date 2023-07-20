from opentrons import protocol_api

# metadata
metadata = {
    'protocolName': 'JG045 Prep',
    'author': 'John Goertz <jgoertz@ic.ac.uk>',
    #'description': '',
    'apiLevel': '2.8'
}

# protocol run function. the part after the colon lets your editor know
# where to look for autocomplete suggestions
def run(protocol: protocol_api.ProtocolContext):

    # labware
    tiprack1 = protocol.load_labware('opentrons_96_filtertiprack_20ul', '1')
    tiprack2 = protocol.load_labware('opentrons_96_filtertiprack_200ul', '2')
    bulk_reagents = protocol.load_labware('opentrons_24_tuberack_nest_2ml_snapcap', '3', label='Bulk Reagents')
    # A3, B3: 2 mL tubes with 1mL Taqman Fast Advanced Master Mix
    # A6: 2 mL tube with 1mL DI

    P2 = protocol.load_labware('corning_96_wellplate_360ul_flat', '4', label='P2')
    P3 = protocol.load_labware('corning_96_wellplate_360ul_flat', '5', label='P3')
    pcr_plate = protocol.load_labware('microamp_384_wellplate_30ul', '6', label='PCR plate')

    tiprack3 = protocol.load_labware('opentrons_96_filtertiprack_20ul', '7')
    tiprack4 = protocol.load_labware('opentrons_96_filtertiprack_20ul', '8')

    master_mix = protocol.load_labware('opentrons_15_tuberack_falcon_15ml_conical', '9', label='Master Mix')
    # A1: 15 mL Falcon tube for assembling master master mix

    # pipettes
    p20 = protocol.load_instrument( 'p20_single_gen2', 'left', tip_racks=[tiprack1,tiprack3,tiprack4])
    p300 = protocol.load_instrument( 'p300_single_gen2', 'right', tip_racks=[tiprack2])

    ############################################################################
    ## Assemble master master mix
    ############################################################################

    # Water
    p300.transfer(788.9, bulk_reagents['A6'], master_mix['A1'])

    # Forward primer
    p20.transfer(28.2, P2['A12'], master_mix['A1'], new_tip='always')

    # Reverse primer
    p20.transfer(28.2, P2['B12'], master_mix['A1'], new_tip='always')

    # Probe
    p300.transfer(56.4, P2['C12'], master_mix['A1'], new_tip='always')

    # Master Mix
    p300.transfer(563.5, [bulk_reagents[well] for well in ['A3','B3']], master_mix['A1'],
                  new_tip='always', mix_after=(5,200))

    #p300.pick_up_tip()
    #p300.mix(3,200,master_mix['A1'])
    #p300.drop_tip()

    ############################################################################
    ## Define target sources and destinations
    ############################################################################

    source_rows = [chr(ord('B')+i) for i in range(7)]
    top_rows = [chr(ord('A')+i) for i in range(7)]
    bottom_rows = [chr(ord('J')+i) for i in range(7)]

    targets = {
        'BP280_GC30': {
            'source':[P2[row+'8'] for row in source_rows],
            'destination':[[pcr_plate[row+'16'],pcr_plate[row+'17']] for row in top_rows],
        },
        'BP500_GC40': {
            'source':[P2[row+'9'] for row in source_rows],
            'destination':[[pcr_plate[row+'19'],pcr_plate[row+'20']] for row in top_rows],
        },
        'BP500_GC60': {
            'source':[P2[row+'10'] for row in source_rows],
            'destination':[[pcr_plate[row+'16'],pcr_plate[row+'17']] for row in bottom_rows],
        },
        'BP280_GC70': {
            'source':[P3[row+'10'] for row in source_rows],
            'destination':[[pcr_plate[row+'13'],pcr_plate[row+'14']] for row in top_rows],
        },
        'BP160_GC80': {
            'source':[P3[row+'9'] for row in source_rows],
            'destination':[[pcr_plate[row+'10'],pcr_plate[row+'11']] for row in top_rows],
        },
        'BP160_GC60': {
            'source':[P3[row+'8'] for row in source_rows],
            'destination':[[pcr_plate[row+'7'],pcr_plate[row+'8']] for row in top_rows],
        },
        'BP160_GC20': {
            'source':[P3[row+'7'] for row in source_rows],
            'destination':[[pcr_plate[row+'4'],pcr_plate[row+'5']] for row in top_rows],
        },
        'BP160_GC10': {
            'source':[P3[row+'6'] for row in source_rows],
            'destination':[[pcr_plate[row+'1'],pcr_plate[row+'2']] for row in top_rows],
        },
        'BP50_GC75': {
            'source':[P3[row+'5'] for row in source_rows],
            'destination':[[pcr_plate[row+'13'],pcr_plate[row+'14']] for row in bottom_rows],
        },
        'BP50_GC60': {
            'source':[P3[row+'4'] for row in source_rows],
            'destination':[[pcr_plate[row+'10'],pcr_plate[row+'11']] for row in bottom_rows],
        },
        'BP50_GC25': {
            'source':[P3[row+'3'] for row in source_rows],
            'destination':[[pcr_plate[row+'7'],pcr_plate[row+'8']] for row in bottom_rows],
        },
        'BP30_GC70': {
            'source':[P3[row+'2'] for row in source_rows],
            'destination':[[pcr_plate[row+'4'],pcr_plate[row+'5']] for row in bottom_rows],
        },
        'BP30_GC30': {
            'source':[P3[row+'1'] for row in source_rows],
            'destination':[[pcr_plate[row+'1'],pcr_plate[row+'2']] for row in bottom_rows],
        },
    }

    NTC = {'destination':[[pcr_plate[row+'22'],pcr_plate[row+'23']] for row in bottom_rows]}

    pcr_plate_wells = [
        well for tar_wells in targets.values() for row in tar_wells['destination'] for well in row
    ] + [
        well for row in NTC['destination'] for well in row
    ]


    ############################################################################
    ## Fill PCR plate
    ############################################################################


    p20.well_bottom_clearance.aspirate = 2
    p20.well_bottom_clearance.dispense = 2

    # Distribute master mix
    p20.distribute(9, master_mix['A1'], pcr_plate_wells,
                   #disposal_volume=2,
                   new_tip='once',
                   touch_tip=True,
                   )

    # Add targets
    for tar in targets:
        for src,dsts in zip(targets[tar]['source'], targets[tar]['destination']):
            # Use one tip per target source well
            p20.pick_up_tip()

            # Aspirate 1 uL per destination well
            vol = 1*len(dsts)
            p20.aspirate(vol, src)

            ## limit z axis to 10 mm/s
            #p20.default_speed = 10
            ## slowly retract pipette tip from well
            #p20.move_to(src.top(z=+5))
            ## return speed to default
            #p20.default_speed = 400

            # Dispense 1 uL into each destination well
            for dst in dsts:
                p20.dispense(1, dst)
            # Mix each destination well, 3 x 5uL
            for dst in dsts[::-1]:
                p20.mix(3, 5, dst)

            p20.drop_tip()
